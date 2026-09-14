"""Independent Gregorian date features after calendar-to-grid alignment."""
from datetime import date
import numpy as np


def features(dates):
    if not isinstance(dates,(list,tuple)) or not dates:
        raise ValueError('Expected nonempty ISO date strings')
    try:
        parsed=[date.fromisoformat(value) for value in dates]
    except (TypeError,ValueError) as error:
        raise ValueError('Expected valid ISO dates') from error
    minimum=min(value.year for value in parsed)
    if max(value.year for value in parsed)-minimum>127:
        raise ValueError('Relative year exceeds int8 range')
    columns=[(value.day,value.isocalendar().week,value.month,value.year-minimum,
              (value.day+6)//7,value.weekday(),int(value.weekday()>=5)) for value in parsed]
    result=np.array(columns,dtype=np.int8)
    return {name:result[:,i].copy() for i,name in enumerate(
        ('day','iso_week','month','relative_year','week_of_month','weekday','weekend'))}
