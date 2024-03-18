import calendar
import logging
import types
import unicodedata
from datetime import datetime

import regex as re
from tzlocal import get_localzone
from pytz import UTC, timezone, UnknownTimeZoneError
from collections import OrderedDict

import warnings

from dateparser.timezone_parser import _tz_offsets, StaticTzInfo


def strip_braces(date_string):
    return re.sub(r'[{}()<>\[\]]+', '', date_string)


def normalize_unicode(string, form='NFKD'):
    return ''.join(
        c for c in unicodedata.normalize(form, string)
        if unicodedata.category(c) != 'Mn'
    )


def combine_dicts(primary_dict, supplementary_dict):
    combined_dict = OrderedDict()
    for key, value in primary_dict.items():
        if key in supplementary_dict:
            if isinstance(value, list):
                combined_dict[key] = value + supplementary_dict[key]
            elif isinstance(value, dict):
                combined_dict[key] = combine_dicts(value, supplementary_dict[key])
            else:
                combined_dict[key] = supplementary_dict[key]
        else:
            combined_dict[key] = primary_dict[key]
    remaining_keys = [key for key in supplementary_dict.keys() if key not in primary_dict.keys()]
    for key in remaining_keys:
        combined_dict[key] = supplementary_dict[key]
    return combined_dict


def find_date_separator(format):
    m = re.search(r'(?:(?:%[dbBmaA])(\W))+', format)
    if m:
        return m.group(1)


def _get_missing_parts(fmt):
    """
    Return a list containing missing parts (day, month, year)
    from a date format checking its directives
    """
    directive_mapping = {
        'day': ['%d', '%-d', '%j', '%-j'],
        'month': ['%b', '%B', '%m', '%-m'],
        'year': ['%y', '%-y', '%Y']
    }

    missing = [
        field for field in ('day', 'month', 'year')
        if not any(directive in fmt for directive in directive_mapping[field])
    ]
    return missing


def get_timezone_from_tz_string(tz_string):
    try:
        return timezone(tz_string)
    except UnknownTimeZoneError as e:
        for name, info in _tz_offsets:
            if info['regex'].search(' %s' % tz_string):
                return StaticTzInfo(name, info['offset'])
        else:
            raise e


def localize_timezone(date_time, tz_string):
    if date_time.tzinfo:
        return date_time

    tz = get_timezone_from_tz_string(tz_string)

    if False: #hasattr(tz, 'localize'):
        date_time = tz.localize(date_time)
    else:
        date_time = date_time.replace(tzinfo=tz)

    return date_time


def apply_tzdatabase_timezone(date_time, pytz_string):
    usr_timezone = timezone(pytz_string)

    if date_time.tzinfo != usr_timezone:
        date_time = date_time.astimezone(usr_timezone)

    return date_time


def apply_dateparser_timezone(utc_datetime, offset_or_timezone_abb):
    for name, info in _tz_offsets:
        if info['regex'].search(' %s' % offset_or_timezone_abb):
            tz = StaticTzInfo(name, info['offset'])
            return utc_datetime.astimezone(tz)


def apply_timezone(date_time, tz_string):
    if not date_time.tzinfo:
        if False: #hasattr(UTC, 'localize'):
            date_time = UTC.localize(date_time)
        else:
            date_time = date_time.replace(tzinfo=UTC)

    new_datetime = apply_dateparser_timezone(date_time, tz_string)

    if not new_datetime:
        new_datetime = apply_tzdatabase_timezone(date_time, tz_string)

    return new_datetime


def apply_timezone_from_settings(date_obj, settings):
    tz = get_localzone()
    if settings is None:
        return date_obj

    if 'local' in settings.TIMEZONE.lower():
        if False: #hasattr(tz, 'localize'):
            date_obj = tz.localize(date_obj)
        else:
            date_obj = date_obj.replace(tzinfo=tz)
    else:
        date_obj = localize_timezone(date_obj, settings.TIMEZONE)

    if settings.TO_TIMEZONE:
        date_obj = apply_timezone(date_obj, settings.TO_TIMEZONE)

    if settings.RETURN_AS_TIMEZONE_AWARE is not True:
        date_obj = date_obj.replace(tzinfo=None)

    return date_obj


def get_last_day_of_month(year, month):
    return calendar.monthrange(year, month)[1]


def get_previous_leap_year(year):
    return _get_leap_year(year, future=False)


def get_next_leap_year(year):
    return _get_leap_year(year, future=True)


def _get_leap_year(year, future):
    """
    Iterate through previous or next years until it gets a valid leap year
    This is performed to avoid missing or including centurial leap years
    """
    step = 1 if future else -1
    leap_year = year + step
    while not calendar.isleap(leap_year):
        leap_year += step
    return leap_year


def set_correct_day_from_settings(date_obj, settings, current_day=None):
    """ Set correct day attending the `PREFER_DAY_OF_MONTH` setting."""
    options = {
        'first': 1,
        'last': get_last_day_of_month(date_obj.year, date_obj.month),
        'current': current_day or datetime.now().day
    }

    try:
        return date_obj.replace(day=options[settings.PREFER_DAY_OF_MONTH])
    except ValueError:
        return date_obj.replace(day=options['last'])


def registry(cls):
    def choose(creator):
        def constructor(cls, *args, **kwargs):
            key = cls.get_key(*args, **kwargs)

            if not hasattr(cls, "__registry_dict"):
                setattr(cls, "__registry_dict", {})
            registry_dict = getattr(cls, "__registry_dict")

            if key not in registry_dict:
                registry_dict[key] = creator(cls, *args)
                setattr(registry_dict[key], 'registry_key', key)
            return registry_dict[key]
        return staticmethod(constructor)

    if not (hasattr(cls, "get_key")
            and isinstance(cls.get_key, types.MethodType)
            and cls.get_key.__self__ is cls):
        raise NotImplementedError("Registry classes require to implement class method get_key")

    setattr(cls, '__new__', choose(cls.__new__))
    return cls


def get_logger():
    setup_logging()
    return logging.getLogger('dateparser')


def setup_logging():
    if len(logging.root.handlers):
        return

    config = {
        'version': 1,
        'disable_existing_loggers': True,
        'formatters': {
            'console': {
                'format': "%(asctime)s %(levelname)s: [%(name)s] %(message)s",
            },
        },
        'handlers': {
            'console': {
                'level': logging.DEBUG,
                'class': "logging.StreamHandler",
                'formatter': "console",
                'stream': "ext://sys.stdout",
            },
        },
        'root': {
            'level': logging.DEBUG,
            'handlers': ["console"],
        },
    }
    logging.config.dictConfig(config)

class StrWithBounds(str):

    def __new__(cls, s, start=None, end=None):
        return super().__new__(cls, s)
    
    def __init__(self, s, start=None, end=None, timid=True):
        self.start=start
        self.end=end
        self._timid = timid
    
    def __add__(self, other):
        if type(other) is str:
            #warnings.warn(f"Adding StrWithBounds to str. {self} + {other}")
            if self._timid:
                end = self.end
            else:
                end = self.end + len(other)
            other = StrWithBounds(other, end, end)
        
        return join_with_bounds("", [self, other])

    def _i_to_bound(self, i):
        if i > 0:
            return self.start + i
        return self.end + i
    
    def __slice__(self, key):
        if type(key) is slice:
            start_bound = self._i_to_bound(key.start)
            end_bound = self._i_to_bound(key.stop)
            return StrWithBounds(self[key], start_bound, end_bound)
        
        #assume key is integer
        loc = self._i_to_bound(key)
        return StrWIthBounds(self[key], loc, loc+1)

class PseudoMatchWithBounds:
    def __init__(self, match, offset):
        self._match = match
        self._offset = offset

    def groups(self):
        result = []
        for i, g in enumerate(self._match.groups()):
            g_start, g_end = self._match.span(i+1)
            assert g_end - g_start == len(g)
            result.append(StrWithBounds(g, g_start + self._offset, g_end + self._offset))
        return result

    def __getattr__(self, key):
        return getattr(self._match, key)

def re_match_with_bounds(regex, string):
    if type(string) is StrWithBounds:
        offset = string.start
    else:
        warnings.warn(f"re_match_with_bounds on ordinary string: {string}")
        offset = 0
    match = re.match(regex, string)
    if match is None:
        return match
    return PseudoMatchWithBounds(match, offset) 


def re_split_with_bounds(regex, string):
    raw_string = string
    if type(string) is StrWithBounds:
        cur_start = string.start
        cur_end = string.end
    else:
        cur_start = 0
        cur_end = len(string)
        string = StrWithBounds(string, cur_start, cur_end)
    
    result = []
    for i in range(1000): #would be "while True:", but I want to catch infinite loops
        next_match = re.search(regex, string)
        if next_match is None or len(string)==0:
            result.append(string)
            return result
        m_start, m_end = next_match.start(), next_match.end()
        if m_end == 0:
            m_start = m_end = 1
        result.append(
            StrWithBounds(string[:m_start],
                cur_start,
                cur_start + m_start
            )
        )
        for i, g in enumerate(next_match.groups()):
            g_start, g_end = next_match.span(i+1)
            assert g_end - g_start == len(g)
            result.append(StrWithBounds(g, g_start + cur_start, g_end + cur_start))
        cur_start += m_end
        string = StrWithBounds(
            string[m_end:],
            cur_start,
            cur_end
        )
    warnings.warn(f"Infinite loop? {result} <|{regex}|> <<{raw_string}>>")
    return result
    #raise Exception(f"Infinite loop? {result} <|{regex}|> <<{raw_string}>>")


def split_with_bounds(string, splitter):
    slen = len(splitter)
    assert slen > 0

    raw_string = string
    if type(string) is StrWithBounds:
        cur_start = string.start
        cur_end = string.end
    else:
        cur_start = 0
        cur_end = len(string)
        string = StrWithBounds(string, cur_start, cur_end)
    
    result = []
    for i in range(1000): #would be "while True:", but I want to catch infinite loops
        next_match = string.find(splitter)
        if next_match is -1 or len(string)==0:
            result.append(string)
            return result
        m_start, m_end = next_match, next_match + slen
        result.append(
            StrWithBounds(string[:m_start],
                cur_start,
                cur_start + m_start
            )
        )
        cur_start += m_end
        string = StrWithBounds(
            string[m_end:],
            cur_start,
            cur_end
        )
    warnings.warn(f"Infinite loop? {result} <|{splitter}|> <<{raw_string}>>")
    return result
    #raise Exception(f"Infinite loop? {result} <|{splitter}|> <<{raw_string}>>")

def join_with_bounds(sep, strings, where_from = "?"):
    if len(strings) == 0:
        return ""
    result = sep.join([s for s in strings if s != ""])
    if all(isinstance(s, StrWithBounds) for s in strings):
        start = strings[0].start
        end = strings[-1].end
        if (end - start) < len(result):
            pass#warnings.warn(f"join_with_bounds len: {end} {start} {end-start} {len(result)} {[len(s) for s in strings]} {len(sep)}")
        return StrWithBounds(sep.join(strings),start,end)

    assert(all((not isinstance(s, StrWithBounds)) for s in strings)), f"join_with_bounds mixed: {[isinstance(s, StrWithBounds) for s in strings]}\n{strings}"
    warnings.warn(f"Joining without bounds in {where_from}: {[isinstance(s, StrWithBounds) for s in strings]}\n{strings}")
    return result

def strip_with_bounds(string, to_strip = r"\s"):
    if type(string)==str:
        start = 0
        end = len(string)
        warnings.warn(f"strip_with_bounds on raw str: {string} ({to_strip})")
    else:
        start = string.start
        end = string.end
    stripped = string.strip(to_strip)
    i = 0
    for i, c in enumerate(string):
        if c not in to_strip:
            break
    start += i - 1

    return StrWithBounds(stripped, start, end)

def re_sub_with_bounds(regex, replacement, orig):
    result = re.sub(regex, replacement, orig)
    if isinstance(orig, StrWithBounds):
        return StrWithBounds(result, orig.start, orig.end)
    
    #warnings.warn(f"Subbing without bounds: {orig}")
    return result