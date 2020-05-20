import numpy as np

from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment

from terran.face.detection import Detection, face_detection


def linear_assignment(cost_matrix):
    """Implement the linear assignment as in Scikit Learn v0.21"""
    return np.transpose(np.asarray(linear_sum_assignment(cost_matrix)))


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

    Parameters
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

    Parameters
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


class KalmanTracker:
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

        self.hits = 0
        self.time_since_update = 0

        self.id = KalmanTracker.count
        KalmanTracker.count += 1

    def update(self, face):
        """Updates the state vector with observed face.

        Parameters
        ----------
        face : dict
            Face to update the tracker with, an observation that's already been
            assigned to the predicted trajectory of the tracker.

        """
        self.time_since_update = 0
        self.hits += 1
        self.kf.update(corners_to_center(face['bbox']))

    def predict(self):
        """Advances the state vector and returns the predicted bounding box
        estimate.

        Returns
        -------
        np.ndarray
            Predicted trajectory of the tracker, in (x_min, y_min, x_max,
            y_max) format.

        """
        # If the size of the bounding box is negative, nullify the velocity.
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0

        self.kf.predict()
        self.time_since_update += 1

        return center_to_corners(self.kf.x)


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
        Matched, unmatched faces and unmatched trackers.

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

    Implements a multiple object tracker based on `Simple Online and Realtime
    Tracking`_ focused on building *face tracks* out of detections (thus
    performing tracking-by-detection).

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

    def __init__(self, max_age=1, min_hits=3, return_unmatched=False):
        """Initialize SORT instance with its configuration.

        The recommended way of setting the time-based parameters is with
        respect to the framerate of the video.

        Parameters
        ----------
        max_age : int
            Maximum number of steps after which an unmatched tracker is going
            to be destroyed.
        min_hits : int
            Minimum number of steps until a tracker is considered confirmed
            (and thus, its identity returned in faces).
        return_unmatched : bool
            Whether to return faces with no track attached (default: False).

        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.return_unmatched = return_unmatched

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
            Same list of face dicts, but with an extra `track` field specifying
            the identity of the face. This field will either be `None` if the
            face wasn't matched to any underlying track, or an `int` if it was.

            If `self.return_unmatched` is `False`, all faces will have a
            `track`, or else they'll get filtered.

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

        # Faces to return, augmented with the track ID, whenever available.
        augmented_faces = []

        # Update matched trackers with assigned detections.
        for track_idx, track in enumerate(self.trackers):
            if track_idx not in unmatched_tracks:
                face_idx = int(
                    matched[np.where(matched[:, 1] == track_idx)[0], 0]
                )
                track.update(faces[face_idx])

                # Add the track ID to the `track` field only if the track has
                # been confirmed.
                track_id = track.id if (
                    track.hits >= self.min_hits
                    or self.frame_count <= self.min_hits
                ) else None

                augmented_faces.append(
                    {'track': track_id, **faces[face_idx]}
                )

        # Create and initialize new trackers for unmatched detections.
        for face_idx in unmatched_faces:
            track = KalmanTracker(faces[face_idx])
            self.trackers.append(track)

            # Just created: only case we should return it right away is if
            # there are no minimum amount of hits required.
            track_id = track.id if self.min_hits == 0 else None
            augmented_faces.append(
                {'track': track_id, **faces[face_idx]}
            )

        # Filter out the faces without a confirmed tracker attached.
        if not self.return_unmatched:
            augmented_faces = [
                face for face in augmented_faces
                if face['track'] is not None
            ]

        # Finally, clean up dead tracks.
        self.trackers = [
            track for track in self.trackers
            if track.time_since_update <= self.max_age
        ]

        return augmented_faces


class FaceTracking:
    """Object for performing face tracking.

    This object is meant to be used as a substitute to a `Detection` object,
    behaving exactly the same way except for having an extra `track` field in
    the face dictionaries.

    The object will only encapsulate and call the detector and tracker objects
    used, offer a :meth:`__call__`-based interface. That is, it's simply a
    container for the main :class:`Sort` class.

    """

    def __init__(self, detector=None, tracker=None):
        self.detector = detector
        self.tracker = tracker

    def __call__(self, frames):
        """Performs face tracking on `images`.

        The face detection itself will be done by the `self.detector` object,
        while the tracking by the `self.tracker` object.

        Parameters
        ----------
        frames : list of numpy.ndarray or numpy.ndarray
            Frames to perform face tracking on.

        Returns
        -------
        list of list of dicts, or list dict
            List of dictionaries containing face data for a single image, or a
            list of these entries thereof.

            Each entry is of the form::

                {
                    'bbox': [x_min, y_min, x_max, y_max],
                    'landmarks': ...,  # Array of shape (5, 2).
                    'track': ...,  # Either an `int` or `None`.
                    'score': ... # Confidence score.
                }

        """
        expanded = False
        if not isinstance(frames, list) and len(frames.shape) == 3:
            expanded = True
            frames = frames[0]

        faces_per_frame = []
        detections_per_frame = self.detector(frames)
        for frame, detections in zip(frames, detections_per_frame):
            faces_per_frame.append(
                self.tracker.update(detections)
            )

        return faces_per_frame[0] if expanded else faces_per_frame


def face_tracking(
    *, video=None, max_age=None, min_hits=None, detector=None,
    return_unmatched=False,
):
    """Default entry point to face tracking.

    This is a factory for an underlying :class:`FaceTracking` instance,
    which will be tasked with keeping the state of the different identities
    available.

    Once you create it, you can treat the resulting object as if it was an
    instance of the :class:`terran.face.detection.Detection` class, but focused
    on working in same-size batches of frames, and returning an additional
    field on the faces corresponding to the identity, or track.

    The tracking utilities provided focus on filtering observations *only*.
    No smoothing nor interpolation will be performed, so the results you
    obtained can be traced back to detections of the detector passed on to it.
    This is meant as a building block from which to do more detailed face
    recognition over videos.

    Parameters
    ----------
    video : terran.io.reader.Video
        Video to derive `max_age` and `min_hits` from. The first value will be
        one video of the second, while the latter will be 1/5th of a second.

        When those values are specified as well, they'll have precedence.
    max_age : int
        Maximum number of frames to keep identities around for after no
        appearance.
    min_hits : int
        Minimum number of observations required for an identity to be returned.

        For instance, if `min_hits` is 6, it means that only after a face is
        detected six times will it be returned on prediction. This is, in
        essence, adding *latency* to the predictions. So consider decreasing
        this value if you care more about latency than any possible noise you
        may get due to short-lived faces.

        You can also get around this latency by specifying `return_unmatched`
        value of `True`, but in that case, returned faces will *not* have an
        identity associated.
    detector : terran.face.detection.Detection
        Face detector to get observations from. Default is using Terran's
        default face detection class.
    return_unmatched : boolean
        Whether to return observations (faces) that don't have a matched
        identity or not.

    """
    # Default values for SORT assume a 30 fps video.
    max_age_ = 30
    min_hits_ = 6

    # If we receive a video or any of the parameters are specified, substitute
    # the defaults.
    if video is not None:
        max_age_ = video.framerate
        min_hits_ = video.framerate // 5

    if max_age is None:
        max_age = max_age_
    if min_hits is None:
        min_hits = min_hits_

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
        return_unmatched=return_unmatched,
    )

    return FaceTracking(detector=detector, tracker=sort)
