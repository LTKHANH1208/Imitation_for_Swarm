from types import SimpleNamespace

from box_pusher_demonstrations.sync_utils import StampedCache


def msg(stamp_ns, label):
    sec, nanosec = divmod(stamp_ns, 1_000_000_000)
    header = SimpleNamespace(stamp=SimpleNamespace(sec=sec, nanosec=nanosec))
    return SimpleNamespace(header=header, label=label)


def test_latest_at_or_before_uses_past_not_future():
    cache = StampedCache(maxlen=5)
    cache.append(msg(100, "a"))
    cache.append(msg(200, "b"))
    cache.append(msg(300, "c"))
    selected = cache.latest_at_or_before(250, 1e-6)
    assert selected is not None
    assert selected.message.label == "b"
    assert selected.stamp_ns == 200


def test_stale_sample_is_rejected():
    cache = StampedCache(maxlen=5)
    cache.append(msg(100, "old"))
    assert cache.latest_at_or_before(500_000_200, 0.1) is None


def test_out_of_order_messages_are_sorted():
    cache = StampedCache(maxlen=5)
    cache.append(msg(300, "c"))
    cache.append(msg(100, "a"))
    cache.append(msg(200, "b"))
    selected = cache.latest_at_or_before(250, 1e-6)
    assert selected is not None
    assert selected.message.label == "b"


def test_zero_timestamp_is_ignored():
    cache = StampedCache(maxlen=5)
    cache.append(msg(0, "zero"))
    assert len(cache) == 0
