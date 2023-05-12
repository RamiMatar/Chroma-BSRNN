hparams_def = {
            "lr" : 3e-4,
            "bandwidths": "15,15,15,15,20,20,20,30,30,30,30,30,30,30,30,50,50,50,50,70,70,100,100,125",
            'n_fft' : 2048,
            'n_mels' : 40,
            'hop_length' : 512,
            'win_length' : 2048,
            'N' : 128,
            'T' : 517,
            'blstm_hidden_size' : 256,
            'num_layers' : 12,
            'sample_rate' : 44100
          }

hparams_chroma = {
            "lr" : 1e-3,
            "bandwidths": "15,15,15,15,20,20,20,30,30,30,30,30,30,30,30,50,50,50,50,70,70,100,100,125",
            'n_fft' : 2048,
            'n_mels' : 40,
            'hop_length' : 512,
            'win_length' : 2048,
            'N' : 128,
            'T' : 517,
            'blstm_hidden_size' : 256,
            'num_bands_layers' : 4,
            'num_chroma_layers' : 2,
            'num_combined_layers' : 2,
            'sample_rate' : 44100,
            'coeff1' : 1,
            'coeff2' : 0
          }
