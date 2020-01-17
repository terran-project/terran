import numpy as np

from filterpy.kalman import KalmanFilter
from sklearn.utils.linear_assignment_ import linear_assignment

from terran.face.detection import Detection, face_detection


def iou(bbox_1, bbox_2):
    """Computes intersection over union between two bounding boxes.

    Parameters
    ----------
    bbox_1 : np.ndarray
        First bounding box, of the form (x_min, y_min, x_max, y_max).
    bbox_2 : np.ndarray
        Second bounding box, of the form (x_min, y_min, x_max, y_max).

    Returns
    -------
    float
        Intersection over union value between both bounding boxes.

    """
    x_min = np.maximum(bbox_1[0], bbox_2[0])
    y_min = np.maximum(bbox_1[1], bbox_2[1])
    x_max = np.minimum(bbox_1[2], bbox_2[2])
    y_max = np.minimum(bbox_1[3], bbox_2[3])
    width = np.maximum(0.0, x_max - x_min)
    height = np.maximum(0.0, y_max - y_min)
    intersection = width * height

    return (
        intersection
    ) / (
        (bbox_1[2] - bbox_1[0]) * (bbox_1[3] - bbox_1[1])
        + (bbox_2[2] - bbox_2[0]) * (bbox_2[3] - bbox_2[1])
        - intersection
    )


def corners_to_center(bbox):
    """Changes bounding box from corner-based specification to center-based.

    Paramters
    ---------
    bbox : np.ndarray
        Bounding box of the form (x_min, y_min, x_max, y_max).

    Returns
    -------
    np.ndarray
        Same bounding box, but of the form (x, y, area, ratio).

    """
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]

    x = bbox[0] + width / 2.0
    y = bbox[1] + height / 2.0

    area = width * height
    ratio = width / height

    return np.array([
        x, y, area, ratio
    ]).reshape((4, 1))


def center_to_corners(bbox):
    """Changes bounding box from corner-based specification to center-based.

    Paramters
    ---------
    bbox : np.ndarray
        Bounding box of the form (x, y, area, ratio).

    Returns
    -------
    np.ndarray
        Same bounding box, but of the form (x_min, y_min, x_max, y_max).

    """
    width = np.sqrt(bbox[2] * bbox[3])
    height = bbox[2] / width

    return np.concatenate([
        bbox[0] - width / 2.0,
        bbox[1] - height / 2.0,
        bbox[0] + width / 2.0,
        bbox[1] + height / 2.0
    ])


class KalmanBoxTracker:
    """Tracker for individual face.

    Maintains an internal state by way of a Kalman filter applied to the face's
    bounding boxes.

    The Kalman filter used tracks the bounding box's center point, scale and
    ratio, and assumes a constant velocity of the box, and a constant ratio for
    the faces: we don't try to estimate the velocity of the ratio, as it's
    bound to be incorrect in a very short time.
    """

    # Count of all trackers instantiated, to keep ascending names for tracks.
    count = 0

    def __init__(self, face):
        """Initializes a tracker using an initial detected face.

        Parameters
        ----------
        face : dict
            Face to initialize tracker to, as returned by a `Detection`
            instance.

        """
        # 4 measurements plus 3 velocities. We don't keep a velocity for the
        # ratio of the bounding box.
        self.kf = KalmanFilter(dim_x=7, dim_z=4)

        self.kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1],
        ])

        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
        ])

        self.kf.R[2:, 2:] *= 10.0

        # Give high uncertainty to the unobservable initial velocities.
        self.kf.P[4:, 4:] *= 1000.0
        self.kf.P *= 10.0

        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        self.kf.x[:4] = corners_to_center(face['bbox'])

        self.time_since_update = 0
        self.id = KalmanBoxTracker.count

        KalmanBoxTracker.count += 1

        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

    def update(self, face):
        """Updates the state vector with observed face."""
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(corners_to_center(face['bbox']))

    def predict(self):
        """Advances the state vector and returns the face with predicted
        bounding box estimate.
        """
        # If the size of the bounding box is negative, nullify the velocity.
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0

        self.kf.predict()

        self.age += 1

        # TODO: I think it's better this way; we want to show face even if it
        # wasn't detected at this particular frame.
        # if self.time_since_update > 0:
        #     self.hit_streak = 0
        self.time_since_update += 1

        self.history.append(center_to_corners(self.kf.x))

        return self.history[-1]

    def get_state(self):
        """Returns the face with the current bounding box estimate."""
        face = {
            'bbox': center_to_corners(self.kf.x),
            'name': self.id,
            'text': f'#{self.id}',
        }

        return face


def associate_detections_to_trackers(faces, trackers, iou_threshold=0.3):
    """Assigns detections to tracked faces.

    Parameters
    ----------
    faces : list
        Observed faces, as returned by a Detection class.
    trackers : np.ndarray of size (T,)
        Positions for the `T` existing trackers.
    iou_threshold : float
        Threshold of IoU value for considering two boxes a match.

    Returns
    -------
    list of np.ndarray
        Matches, unmatched faces and unmatched trackers.

    """

    if not len(trackers):
        return (
            np.empty((0, 2), dtype=int),
            np.arange(len(faces)),
            np.empty((0, 5), dtype=int),
        )

    iou_matrix = np.zeros(
        (len(faces), len(trackers)),
        dtype=np.float32
    )
    for face_idx, face in enumerate(faces):
        for track_idx, track in enumerate(trackers):
            iou_matrix[face_idx, track_idx] = iou(face['bbox'], track)

    # Use the hungarian method to get least-cost assignment. Returns an array
    # of size (min(len(faces), len(trackers)), 2), with the (row, col) indices
    # of the matches.
    matched_indices = linear_assignment(-iou_matrix)

    unmatched_faces = []
    for face_idx, face in enumerate(faces):
        if face_idx not in matched_indices[:, 0]:
            unmatched_faces.append(face_idx)

    unmatched_trackers = []
    for track_idx, track in enumerate(trackers):
        if track_idx not in matched_indices[:, 1]:
            unmatched_trackers.append(track_idx)

    # Filter out matched with low IOU.
    matches = []
    for face_idx, track_idx in matched_indices:
        if iou_matrix[face_idx, track_idx] < iou_threshold:
            unmatched_faces.append(face_idx)
            unmatched_trackers.append(track_idx)
        else:
            matches.append(
                np.array([face_idx, track_idx], dtype=int)
            )

    if not matches:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.stack(matches)

    return (
        matches, np.array(unmatched_faces), np.array(unmatched_trackers)
    )


class Sort:
    """SORT tracker for very simple, appearence-agnostic tracking.

    Implements a tracker based on `Simple Online and Realtime Tracking`_
    focused on building *face tracks* out of detections (thus performing
    tracking-by-detection).

    The tracking performed by this class has only one objective: attach an
    identity to every detection passed to it, or `None` if no identity was
    found or yet constructed for it.

    This means the class will not:
    * Smooth out the bounding boxes: observations are returned as-is.
    * Interpolate a face whenever there's a missing observation in between.

    If you need to do get the smoothed out observations, your best bet is to
    modify the `self.update()` method to return the tracker predictions instead
    of the observations. We might make it a configurable at a later point.

    .. _Simple Online and Realtime Tracking: https://arxiv.org/abs/1602.00763

    """

    def __init__(self, max_age=1, min_hits=3, keep_showing_for=5):
        self.max_age = max_age
        self.min_hits = min_hits
        self.keep_showing_for = keep_showing_for
        self.trackers = []
        self.frame_count = 0

    def update(self, faces):
        """Update the tracker with new faces.

        This function should be called on every frame, even if no faces are
        detected.

        Parameters
        ----------
        faces : list
            List of faces dicts, as returned by a `Detection` instance.

        Returns
        -------
        list of dicts
            List of faces, similar to input, but only for the tracked faces.

        """
        self.frame_count += 1

        # Get predicted locations from existing trackers.
        to_delete = []
        tracks = np.zeros((len(self.trackers), 4))
        for track_idx, track in enumerate(tracks):

            position = self.trackers[track_idx].predict()
            track[:] = position

            # A tracker might have gone to infinity in its extrapolation.
            if np.any(np.isnan(position)):
                to_delete.append(track_idx)

        tracks = np.ma.compress_rows(
            np.ma.masked_invalid(tracks)
        )

        for t in reversed(to_delete):
            self.trackers.pop(t)

        (
            matched, unmatched_faces, unmatched_tracks
        ) = associate_detections_to_trackers(faces, tracks)

        # Update matched trackers with assigned detections.
        for track_idx, track in enumerate(self.trackers):
            if track_idx not in unmatched_tracks:
                face_idx = int(
                    matched[np.where(matched[:, 1] == track_idx)[0], 0]
                )
                track.update(faces[face_idx])

        # Create and initialize new trackers for unmatched detections.
        for face_idx in unmatched_faces:
            track = KalmanBoxTracker(faces[face_idx])
            self.trackers.append(track)

        tracked_faces = []
        for track_idx, track in reversed(list(enumerate(self.trackers))):
            face = track.get_state()

            should_return = (
                track.time_since_update <= self.keep_showing_for
            ) and (
                track.hit_streak >= self.min_hits
                or self.frame_count <= self.min_hits
            )

            if should_return:
                tracked_faces.append(face)

            # Remove dead tracks.
            if track.time_since_update > self.max_age:
                self.trackers.pop(track_idx)

        return tracked_faces


class FaceTracking:

    def __init__(self, detector=None, sort=None):
        self.detector = detector
        self.sort = sort

    def __call__(self, frames):
        expanded = False
        if not isinstance(frames, list) and len(frames.shape) == 3:
            expanded = True
            frames = frames[0]

        faces_per_frame = []
        detections_per_frame = self.detector(frames)
        for frame, detections in zip(frames, detections_per_frame):
            faces_per_frame.append(
                self.sort.update(detections)
            )

        return faces_per_frame[0] if expanded else faces_per_frame


def face_tracking(
    *, video=None, max_age=None, min_hits=None, keep_showing_for=None,
    detector=None,
):

    # Default values for SORT assume a 30 fps video.
    max_age_ = 30
    min_hits_ = 6
    keep_showing_for_ = 5

    # If we receive a video or any of the parameters are specified, substitute
    # the defaults.
    if video is not None:
        max_age_ = video.framerate
        min_hits_ = video.framerate // 5
        keep_showing_for_ = video.framerate // 6

    if max_age is None:
        max_age = max_age_
    if min_hits is None:
        min_hits = min_hits_
    if keep_showing_for is None:
        keep_showing_for = keep_showing_for_

    # Validate that we received a valid detector, or fall back to the default
    # one if none was specified.
    if detector is None:
        detector = face_detection
    elif not isinstance(detector, Detection):
        raise ValueError(
            '`detector` must be an instance of `terran.face.Detection`.'
        )

    sort = Sort(
        max_age=video.framerate,
        min_hits=video.framerate // 5,
        keep_showing_for=video.framerate // 6,
    )

    return FaceTracking(detector=detector, sort=sort)
