from fastai.vision import *
import pre
import resample

import os

# The time series that you would get are such that the difference between two rows is 15 minutes.
# This is a global number that we used to prepare the data, so you would need it for different purposes.
DATA_RESOLUTION_MIN = 15


def normalize_time(series):
    # 1440 minutes in a day
    normalized = (series.hour * 60 + series.minute) / 1440
    return normalized


def build_features(cgm, meals):
    meals = resample.resample_meals(cgm, meals, 15)
    meals = pd.concat((meals, cgm), axis=1)
    meals['time'] = normalize_time(meals.index.get_level_values('Date'))
    cgm, y = pre.build_cgm(cgm)
    return cgm, meals, y


def get_data(data_dir):
    cgm, meals = pre.get_dfs(data_dir)
    return build_features(cgm, meals)


class ContData(Dataset):
    def __init__(self, cgm, meals, y):
        self.cgm = cgm
        self.meals = meals
        self.y = y

    def __len__(self):
        return len(self.cgm)

    def __getitem__(self, i):
        index = self.meals.index.get_loc(self.cgm.index[i])
        values = self.meals[index - 48:index + 1].values
        target = self.y.iloc[i].values
        x, y = torch.tensor(values, dtype=torch.float), torch.tensor(target, dtype=torch.float)
        return x, y


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Linear(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input)
        output = embedded
        output, hidden = self.gru(output[None], hidden)
        return output[0], hidden

    def initHidden(self, bs, device):
        return torch.zeros(1, bs, self.hidden_size, device=device)


MAX_LENGTH = 49


class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size=1, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Linear(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(self.attn(torch.cat((embedded, hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights[:, None], encoder_outputs)

        output = torch.cat((embedded, attn_applied[:, 0]), 1)
        output = self.attn_combine(output)

        output = F.relu(output)
        output, hidden = self.gru(output[None], hidden)

        output = self.out(output[0])
        return output, hidden, attn_weights

    def initHidden(self, bs, device):
        return torch.zeros(1, bs, self.hidden_size, device=device)


class Seq2Seq(Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.encoder = EncoderRNN(input_size, hidden_size)
        self.decoder = AttnDecoderRNN(hidden_size)

    def forward(self, input):
        device = input.device
        bs = input.shape[0]
        input = input.transpose(0, 1)

        encoder_hidden = self.encoder.initHidden(bs, device)
        encoder_outputs = input.new_zeros(bs, MAX_LENGTH, self.encoder.hidden_size)

        for ei in range(input.shape[0]):
            encoder_output, encoder_hidden = self.encoder(input[ei], encoder_hidden)
            encoder_outputs[:, ei] = encoder_output

        decoder_input = input.new_zeros(bs, 1)
        decoder_hidden = encoder_hidden

        out = []
        for di in range(8):
            decoder_output, decoder_hidden, decoder_attention = self.decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            out.append(decoder_output)
            decoder_input = decoder_output.detach()

        out = torch.cat(out, dim=1)
        return out


class PredClbk(Callback):

    def __init__(self, length):
        super().__init__()
        self.pred = np.empty((length, 8))
        self.start = 0

    def on_batch_end(self, last_output, last_target, **kwargs):
        values = last_output.cpu().numpy()
        end = self.start + len(values)
        self.pred[self.start:end] = values
        self.start = end


root = Path(__file__).resolve().parent / 'our_data'
val = root / 'val'


class Predictor(object):
    """
    This is where you should implement your predictor.
    The testing script calls the 'predict' function with the glucose and meals test data which you will need in order to
    build your features for prediction.
    You should implement this function as you wish, just do not change the function's signature (name, parameters).
    The other functions are here just as an example for you to have something to start with, you may implement whatever
    you wish however you see fit.
    """

    def __init__(self, path2data=''):
        """
        This constructor only gets the path to a folder where the training data frames are.
        :param path2data: a folder with your training data.
        """
        self.path2data = path2data

        train_data = get_data(root)
        val_data = get_data(val)

        train_ds = ContData(*train_data)
        val_ds = ContData(*val_data)
        data = DataBunch.create(train_ds, val_ds, bs=512)

        model = Seq2Seq(38, 128)
        self.learner = Learner(data, model, loss_func=nn.MSELoss())
        self.learner.path = root.parent
        self.learner.load('gru-trainval')

    def predict(self, X_glucose, X_meals):
        """
        You must not change the signature of this function!!!
        You are given two data frames: glucose values and meals.
        For every timestamp (t) in X_glucose for which you have at least 12 hours (48 points) of past glucose and two
        hours (8 points) of future glucose, predict the difference in glucose values for the next 8 time stamps
        (t+15, t+30, ..., t+120).

        :param X_glucose: A pandas data frame holding the glucose values in the format you trained on.
        :param X_meals: A pandas data frame holding the meals data in the format you trained on.
        :return: A numpy ndarray, sized (M x 8) holding your predictions for every valid row in X_glucose.
                 M is the number of valid rows in X_glucose (number of time stamps for which you have at least 12 hours
                 of past glucose values and 2 hours of future glucose values.
                 Every row in your final ndarray should correspond to:
                 (glucose[t+15min]-glucose[t], glucose[t+30min]-glucose[t], ..., glucose[t+120min]-glucose[t])
        """
        y_true_index = self.build_features(X_glucose, None)[1].index
        cgm, meals = X_glucose.sort_index(), X_meals.sort_index()
        pre.preprocess(cgm, meals)
        test_data = build_features(cgm, meals)
        test_ds = ContData(*test_data)
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        test_dl = DeviceDataLoader(DataLoader(test_ds, batch_size=128, shuffle=False), device)

        gt = test_data[2]
        clbk = PredClbk(len(gt))
        self.learner.validate(test_dl, callbacks=[clbk])

        pred = clbk.pred * pre.norm_stats['GlucoseValue'][1]
        y = pd.DataFrame(index=gt.index, columns=gt.columns, data=pred)
        y = y.loc[y_true_index]
        return y

    @staticmethod
    def load_data_frame(path):
        """
        Load a pandas data frame in the relevant format.
        :param path: path to csv.
        :return: the loaded data frame.
        """
        return pd.read_csv(path, index_col=[0, 1], parse_dates=['Date'])

    def load_raw_data(self):
        """
        Loads raw data frames from csv files, and do some basic cleaning
        :return:
        """
        self.train_glucose = Predictor.load_data_frame(os.path.join(self.path2data, 'GlucoseValues.csv'))
        self.train_meals = Predictor.load_data_frame(os.path.join(self.path2data, 'Meals.csv'))

        # suggested procedure
        # 1. handle outliers: trimming, clipping...
        # 2. feature normalizations
        # 3. resample meals data to match glucose values intervals
        return

    def build_features(self, X_glucose, X_meals, n_previous_time_points=48, n_future_time_points=8):
        """
        Given glucose and meals data, build the features needed for prediction.
        :param X_glucose: A pandas data frame holding the glucose values.
        :param X_meals: A pandas data frame holding the meals data.
        :param n_previous_time_points:
        :param n_future_time_points:
        :return: The features needed for your prediction, and optionally also the relevant y arrays for training.
        """
        # using X_glucose and X_meals to build the features
        # get the past 48 time points of the glucose
        X = X_glucose.reset_index().groupby('id').apply(Predictor.create_shifts,
                                                        n_previous_time_points=n_previous_time_points).set_index(
            ['id', 'Date'])
        # use the meals data...

        # this implementation of extracting y is a valid one.
        y = X_glucose.reset_index().groupby('id').apply(Predictor.extract_y,
                                                        n_future_time_points=n_future_time_points).set_index(
            ['id', 'Date'])
        index_intersection = X.index.intersection(y.index)
        X = X.loc[index_intersection]
        y = y.loc[index_intersection]
        return X, y

    @staticmethod
    def create_shifts(df, n_previous_time_points=48):
        """
        Creating a data frame with columns corresponding to previous time points
        :param df: A pandas data frame
        :param n_previous_time_points: number of previous time points to shift
        :return:
        """
        for g, i in zip(
                range(DATA_RESOLUTION_MIN, DATA_RESOLUTION_MIN * (n_previous_time_points + 1), DATA_RESOLUTION_MIN),
                range(1, (n_previous_time_points + 1), 1)):
            df['GlucoseValue -%0.1dmin' % g] = df.GlucoseValue.shift(i)
        return df.dropna(how='any', axis=0)

    @staticmethod
    def extract_y(df, n_future_time_points=8):
        """
        Extracting the m next time points (difference from time zero)
        :param n_future_time_points: number of future time points
        :return:
        """
        for g, i in zip(
                range(DATA_RESOLUTION_MIN, DATA_RESOLUTION_MIN * (n_future_time_points + 1), DATA_RESOLUTION_MIN),
                range(1, (n_future_time_points + 1), 1)):
            df['Glucose difference +%0.1dmin' % g] = df.GlucoseValue.shift(-i) - df.GlucoseValue
        return df.dropna(how='any', axis=0).drop('GlucoseValue', axis=1)
