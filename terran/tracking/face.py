from terran.face.detection import Detection, face_detection


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
