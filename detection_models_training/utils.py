import datetime
import time
from collections import defaultdict, deque

import torch


class SmoothedValue:
    """
    Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        """
        Initialize the SmoothedValue object.

        Args:
            window_size (int): The size of the window for calculating smoothed values.
            fmt (str): The format string for displaying values.
        """
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        """
        Update the series with a new value.

        Args:
            value (float): The new value to add.
            n (int): The number of times to add the value.
        """
        self.deque.append(value)
        self.count += n
        self.total += value * n

    @property
    def median(self):
        """Calculate the median of the series."""
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        """Calculate the average of the series."""
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        """Calculate the global average of the series."""
        return self.total / self.count

    @property
    def max(self):
        """Get the maximum value in the series."""
        return max(self.deque)

    @property
    def value(self):
        """Get the most recent value in the series."""
        return self.deque[-1]

    def __str__(self):
        """Return a string representation of the object."""
        return self.fmt.format(
            median=self.median, avg=self.avg, global_avg=self.global_avg, max=self.max, value=self.value
        )


class MetricLogger:
    """
    Log and display metrics during training or evaluation.
    """

    def __init__(self, delimiter="\t"):
        """
        Initialize the MetricLogger object.

        Args:
            delimiter (str): The delimiter for separating values in the log.
        """
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        """
        Update the metrics with new values.

        Args:
            **kwargs: Key-value pairs of metric names and their values.
        """
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        """
        Get a metric by name.

        Args:
            attr (str): The name of the metric.

        Returns:
            SmoothedValue: The requested metric.
        """
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{attr}'")

    def __str__(self):
        """
        Return a string representation of the object.

        Returns:
            str: A string containing all metrics and their values.
        """
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(f"{name}: {str(meter)}")
        return self.delimiter.join(loss_str)

    def add_meter(self, name, meter):
        """
        Add a new metric to the logger.

        Args:
            name (str): The name of the metric.
            meter (SmoothedValue): The metric object.
        """
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        """
        Log metrics for each item in an iterable.

        Args:
            iterable (iterable): The iterable to log.
            print_freq (int): The frequency of logging.
            header (str): The header for the log.
        """
        i = 0
        if not header:
            header = ""
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt="{avg:.4f}")
        data_time = SmoothedValue(fmt="{avg:.4f}")
        space_fmt = ":" + str(len(str(len(iterable)))) + "d"

        # Define log message format
        if torch.cuda.is_available():
            log_msg = self.delimiter.join(
                [
                    header,
                    "[{0" + space_fmt + "}/{1}]",
                    "eta: {eta}",
                    "{meters}",
                    "time: {time}",
                    "data: {data}",
                    "max mem: {memory:.0f}",
                ]
            )
        else:
            log_msg = self.delimiter.join(
                [header, "[{0" + space_fmt + "}/{1}]", "eta: {eta}", "{meters}", "time: {time}", "data: {data}"]
            )
        MB = 1024.0 * 1024.0

        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)

            # Log at the specified frequency or at the end
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(
                        log_msg.format(
                            i,
                            len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                            memory=torch.cuda.max_memory_allocated() / MB,
                        )
                    )
                else:
                    print(
                        log_msg.format(
                            i, len(iterable), eta=eta_string, meters=str(self), time=str(iter_time), data=str(data_time)
                        )
                    )
            i += 1
            end = time.time()

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print(f"{header} Total time: {total_time_str} ({total_time / len(iterable):.4f} s / it)")
