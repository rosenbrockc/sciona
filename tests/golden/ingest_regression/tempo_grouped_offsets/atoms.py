def offset_tt2tdb(seconds):
    return {"offset_seconds": seconds, "variant": "low_order"}


def offset_tt2tdbh(seconds):
    return {"offset_seconds": seconds, "variant": "high_order"}


def tt2tdb_offset(seconds):
    if isinstance(seconds, (list, tuple)):
        return [offset_tt2tdb(value) for value in seconds]
    return offset_tt2tdb(seconds)
