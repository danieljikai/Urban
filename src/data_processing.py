import h5py
import pandas as pd
import datetime
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def check_holidays(timeslots, directory="../TaxiBJ"):
    '''
    check whether those timeslots are holidays in Beijing
    input: a list of timeslots (strings)
    return: an array of 0,1
    1: holiday
    0: not holiday
    '''
    path = "{}/BJ_Holiday.txt".format(directory)
    data = pd.read_csv(path, sep = " ", header = None)
    holidays = set(data[0])
    ans = [int(int(slot[:8]) in holidays) for slot in timeslots]
    print("Timeslot number is: {}\nNumber of holidays is :{}\n".format(len(ans), sum(ans)))
    return np.asarray(ans)

def check_weekends(timeslots):
    '''
    check whether those timeslots are weekends in Beijing
    input: a list of timeslots (strings)
    return: an array of 0,1
    1: weekday
    0: weekend
    '''
    ans = []
    for slot in timeslots:
        date = datetime.datetime.strptime(slot[:8], "%Y%m%d")
        ans.append(int(date.weekday()<5))
    print("Timeslot number is: {}\nNumber of weekdays is :{}\n".format(len(ans), sum(ans)))
    return np.asarray(ans)

def process_meteorol(timeslots, directory="../TaxiBJ"):
    '''
    process the meteorol data
    input: a list of timeslots (strings)
    return: an array of size n*19, n is the length of timeslots
    0-16 columns: weather code
    17 column: 0-1 scaled windspeed
    18 column: 0-1 scaled temperature
    '''
    path = "{}/BJ_Meteorology.h5".format(directory)
    timeslots = set(timeslots)
    data = h5py.File(path)
    date = list(map(lambda x: x.decode('UTF-8'), data["date"]))
    windspeed = np.asarray([v for i,v in enumerate(data["WindSpeed"]) if date[i] in timeslots])
    temperature = np.asarray([v for i,v in enumerate(data["Temperature"]) if date[i] in timeslots])
    weather = np.asarray([v for i,v in enumerate(data["Weather"]) if date[i] in timeslots])
    data.close()
#    print(windspeed.shape)
    # scale the wind speed and temperature data, weather is the categorial data
    windspeed = (windspeed - windspeed.min())/(windspeed.max()-windspeed.min())
    temperature = (temperature - temperature.min())/(temperature.max() - temperature.min())

    ans = np.hstack([weather, windspeed[:,None], temperature[:,None]])
    print("The shape of the meteorol data is: {}\n".format(ans.shape))
    return ans

def remove_incomplete_days(data, timeslots):
    '''
    remove days that don't have complete 48 slots
    input: data and timeslots
    return: filtered data and timeslots
    '''
    complete, incomplete = set(), set()
    i = 0
    n = len(timeslots)
    while i < n:
        if int(timeslots[i][8:]) == 1 and i+47 < n and int(timeslots[i+47][8:]) == 48:
            complete.add(timeslots[i][:8])
            i += 48
        else:
            incomplete.add(timeslots[i][:8])
            i += 1
    print("Complete days: {}\nIncomplete days: {}".format(len(complete), len(incomplete)))

    index = [i for i, slot in enumerate(timeslots) if slot[:8] in complete]
    data = data[index]
    timeslots = [timeslots[i] for i in index]
    return data, timeslots

def timeslot2datetime(timeslot):
    '''
    convert a timeslot to datetime
    input: timeslot (string)
    output: datetime.datetime
    '''
    year, month, day, slot = int(timeslot[:4]), int(timeslot[4:6]), int(timeslot[6:8]), int(timeslot[8:])-1
    hour, minute = slot//2, slot % 2 * 30
    return datetime.datetime(year = year, month = month, day = day, hour = hour, minute = minute)

def datetime2timeslot(t):
    year = t.year
    month = t.month
    day = t.day
    slot = (t.hour * 60 + t.minute)//30 + 1
    return str(year) + str(month).zfill(2) + str(day).zfill(2) + str(slot).zfill(2)

def remove_incomplete_features(data, timeslots, interval = 3):
    '''
    remove timeslots that don't have complete 9 previous points
    0.5 hour*3, 1 day*3, 1 week *3
    input: data and timeslots
    return:
    '''
    offset = datetime.timedelta(minutes = 30)
    timeslots = [timeslot2datetime(slot) for slot in timeslots]
    timeslot_dict = {slot: i for i,slot in enumerate(timeslots)}
    vectors = [range(1,interval + 1),
              [48*i for i in range(1,interval + 1)],
              [48*7*i for i in range(1,interval + 1)]]
    X_final = [[], [], []]
    y_final = []
    slots_final = []
    i = 48*7*interval
    while i < len(timeslots):
        complete = True
        for vector in vectors:
            if not complete:
                break
            complete = all((timeslots[i] - v * offset) in timeslot_dict for v in vector)

        if not complete:
            i += 1
            continue

        X = [[data[timeslot_dict[timeslots[i] - v * offset]] for v in vector] for vector in vectors]
        y = data[timeslot_dict[timeslots[i]]]
        for j in range(3):
            X_final[j].append(np.vstack(X[j]))
        y_final.append(y)
        slots_final.append(timeslots[i])
        i += 1
    print("Timeslots in total: {}\n that have complete features: {}\n".format(len(timeslots),len(slots_final)))
    return np.asarray(X_final), np.asarray(y_final), np.asarray(slots_final)


def scaler(X):
    """
    min max scale to [-1,1]
    """
    ans = np.zeros(X.shape)
    for i in range(X.shape[0]):
        for j in range(0,X.shape[1],2):

            MIN, MAX = X[i,j:j+2].min(), X[i,j:j+2].max()
            if MIN != MAX:
                ans[i][j] = (X[i][j] - MIN) / (MAX - MIN) *2.0 -1
                ans[i][j+1] = (X[i][j+1] - MIN)/(MAX - MIN) *2.0 - 1
    return np.asarray(ans)


def main(directory="../TaxiBJ", scale=False):
    dataset, times = [],[]
    for year in range(13,17):
        path = "{}/BJ{}_M32x32_T30_InOut.h5".format(directory, year)
        datafile = h5py.File(path)
        data, t = datafile["data"], datafile["date"]
        data, t = remove_incomplete_days(data, t)
        data[data<0] = 0
        dataset.append(data)
        times.append(t)
        datafile.close()

    dataset = np.concatenate(dataset)

    times = np.concatenate(times)

    X, y, time = remove_incomplete_features(dataset, times)
    timeslots = [datetime2timeslot(t) for t in time]


    weekday_features = check_weekends(timeslots)
    holiday_features = check_holidays(timeslots, directory = directory)
    meteorol_features = process_meteorol(timeslots, directory = directory)
    social_features = np.hstack([weekday_features[:,None], holiday_features[:,None], meteorol_features])
    print("X_hour shape: {}\nX_day shape: {}\nX_week shape: {}\nX_extra shape: {}\ny shape: {}\ntimeslots shape: {}\n"\
          .format(X[0].shape, X[1].shape, X[2].shape, social_features.shape, y.shape, np.asarray(timeslots).shape))

    if scale:
        return scaler(X[0]), scaler(X[1]), scaler(X[2]),\
            social_features, scaler(y), np.asarray(timeslots)
    else:
        return X[0], X[1], X[2], social_features, y, np.asarray(timeslots)


if __name__ == "__main__":
    DIR = "../TaxiBJ"
    X_hour, X_day, X_week, X_social, y, timeslots = main(directory=DIR, scale=True)
# #    np.save('X_hour', X_hour)
# #    np.save('X_day', X_day)
# #    np.save('X_week', X_week)
# #    np.save('X_extra', X_extra)
# #    np.save('y', y)
# #    np.save('timeslots', timeslots)
# <<<<<<< HEAD
#     X_main = np.concatenate((X_day, X_hour, X_week), axis = 1) # n* 2m* 32 *32
#     data = {"X_hour": X_hour,
#             "X_day": X_day,
#             "X_week": X_week,
#             "X_extra": X_extra,
#             "y": y,
#             "timeslots": timeslots}
#     np.save("temp_data",data)
# =======

    X = np.concatenate((X_day, X_hour, X_week), axis = 1) # n* 2m* 32 *32
    np.save('../data/X', X)
    np.save('../data/X_extra',X_social)
    np.save('../data/y', y)
    np.save('../data/timeslots', timeslots)
#     print(X_hour.shape, X_day.shape, X_week.shape, y.shape)
#     print(X.shape)
# >>>>>>> 3a33c7d6cfd6953ed779b73236a4fa46614604d5
