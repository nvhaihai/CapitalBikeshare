import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_steps):
        super(LSTMModel, self).__init__()
        self.output_steps = output_steps
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_steps)
        self._initialize_weights()

    def _initialize_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        output, (h_n, _) = self.lstm(x)
        h_n = h_n[-1]
        out = self.fc(h_n)
        return out

def print_layer_sizes(model, input_data):
    x = input_data
    print("Input size:", x.size())
    output, (h_n, c_n) = model.lstm(x)
    print("LSTM hidden state size:", h_n.size())
    print("LSTM cell state size:", c_n.size())
    out = model.fc(h_n[-1])
    print("FC layer input size:", h_n[-1].size())
    print("FC layer output size:", out.size())



if __name__=="__main__":
    input_data = torch.randn(32, 96, 16)
    model = LSTMModel(input_size=16, hidden_size=32, output_steps=5)
    print_layer_sizes(model, input_data)