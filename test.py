from data import get_csv_data

train,valid = get_csv_data()
print("Train:" ,train.shape)
print("Valid:" ,valid.shape)