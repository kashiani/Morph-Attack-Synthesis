import cv2

def check_write_video(func):
    """
    Decorator to check if video writing is enabled before executing the function.

    This decorator ensures that the wrapped function is only executed if the `video` attribute
    is initialized and not None.

    :param func: function
        The video-related function to be conditionally executed.

    :returns: function
        Wrapped function that checks `self.video` before execution.
    """
    def inner(self, *args, **kwargs):
        if self.video:
            return func(self, *args, **kwargs)
        else:
            pass
    return inner

class Video:
    """
    A class for creating and managing video files.

    This class facilitates writing images to a video file and ensures proper resource
    management by releasing the video writer when done.

    :param filename: str
        Name of the output video file. If None, video writing is disabled.

    :param fps: float
        Frames per second for the output video.

    :param w: int
        Width of the video frames.

    :param h: int
        Height of the video frames.
    """
    def __init__(self, filename, fps, w, h):
        self.filename = filename

        if filename is None:
            # Disable video writing if no filename is provided
            self.video = None
        else:
            # Initialize the video writer with the specified parameters
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            self.video = cv2.VideoWriter(filename, fourcc, fps, (w, h), True)

    @check_write_video
    def write(self, img, num_times=1):
        """
        Write an image to the video file.

        :param img: numpy.ndarray
            The image to write to the video. Should have the same dimensions as specified
            during initialization.

        :param num_times: int, optional (default=1)
            Number of times to write the same frame to the video.
        """
        for i in range(num_times):
            self.video.write(img[..., :3])

    @check_write_video
    def end(self):
        """
        Finalize the video file and release the video writer.

        Prints a message indicating the video file has been saved.
        """
        print(self.filename + ' saved')
        self.video.release()
