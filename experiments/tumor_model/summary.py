import tensorflow as tf
import bayesflow as bf


class CustomSummary(tf.keras.Model):
    def __init__(self, num_summary=32, num_dense_concat=2):
        super().__init__()    

        self.lstm = tf.keras.layers.LSTM(num_summary // 2)
        self.att_dict = {
            'num_heads': 4,
            'dropout': 0.1,
            'key_dim': 64
        }
        self.dense_dict = {
            'units': 128,
            'activation': 'relu',
            'kernel_regularizer': tf.keras.regularizers.L2(1e-4),
            'bias_regularizer': tf.keras.regularizers.L2(1e-4)
        }
        self.transformer = bf.networks.TimeSeriesTransformer(
            input_dim=3,
            attention_settings=self.att_dict,
            dense_settings=self.dense_dict,
            template_dim=32,
            summary_dim=num_summary
        )  
        dense_layers = []
        for l in range(num_dense_concat):
            dense_layers.append(
                tf.keras.layers.Dense(
                    units=256, 
                    activation='swish', 
                    kernel_regularizer=tf.keras.regularizers.L2(1e-4),
                    bias_regularizer=tf.keras.regularizers.L2(1e-4)
                )
            )
        self.fc = tf.keras.Sequential(dense_layers)
        self.fc.add(tf.keras.layers.Dense(num_summary))

    def call(self, summary_conditions, **kwargs):
        pp_ecmp, growth = summary_conditions        
        out1 = self.lstm(growth, **kwargs)
        out2 = self.transformer(pp_ecmp, **kwargs)        
        out = tf.concat([out1, out2], axis=-1)
        out = self.fc(out, **kwargs)        
        return out
