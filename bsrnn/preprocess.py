import os
import torch
import torchaudio

class PreprocessValidationSet:
    def __init__(self, data_directory, sample_rate, target_directory, last_n_songs, segment_length=6, overlap=0.5):
        self.data_directory = data_directory
        self.sample_rate = sample_rate
        self.target_directory = target_directory
        self.last_n_songs = last_n_songs
        self.segment_length = segment_length
        self.overlap = overlap

        if not os.path.exists(target_directory):
            os.makedirs(target_directory)

    def preprocess(self):
        segment_samples = self.segment_length * self.sample_rate
        step_size = int(segment_samples * (1 - self.overlap))

        # Get the list of songs (subdirectories) in the data directory
        song_list = sorted([d for d in os.listdir(self.data_directory) if os.path.isdir(os.path.join(self.data_directory, d))])

        # Select the last_n_songs subdirectories
        selected_songs = song_list[-self.last_n_songs:]

        for song in selected_songs:
            song_dir = os.path.join(self.data_directory, song)
            audio_files = [os.path.join(song_dir, f) for f in os.listdir(song_dir) if f.endswith('.wav')]
            for audio_filename in audio_files:
                waveform, sr = torchaudio.load(audio_filename)

                # Ensure the waveform has the same sample rate as the target sample rate
                if sr != self.sample_rate:
                    waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sample_rate)(waveform)

                # Zero-pad the waveform if necessary
                total_samples = waveform.shape[-1]
                if total_samples % step_size != 0:
                    padding_size = step_size - (total_samples % step_size)
                    waveform = torch.nn.functional.pad(waveform, (0, padding_size))

                # Extract 6-second segments from the waveform with 50% overlap
                total_samples = waveform.shape[-1]
                segment_idx = 0
                for start_sample in range(0, total_samples - segment_samples + 1, step_size):
                    end_sample = start_sample + segment_samples
                    segment = waveform[:, start_sample:end_sample]

                    # Save the segment in the target directory
                    path_parts = os.path.normpath(audio_filename).split(os.sep)
                    song_name = path_parts[-2]
                    source = path_parts[-1]
                    base_filename = os.path.splitext(source)[0]

                    # Create output directory if it doesn't exist
                    output_subdir = os.path.join(self.target_directory, song_name)
                    if not os.path.exists(output_subdir):
                        os.makedirs(output_subdir)

                    segment_filename = f"{base_filename}_{segment_idx:03d}.wav"
                    segment_path = os.path.join(output_subdir, segment_filename)
                    torchaudio.save(segment_path, segment, self.sample_rate)
                    segment_idx += 1
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Preprocess validation dataset into equal length, overlapping chunks to ensure validation dataset metrics are consistent')
    parser.add_argument('dataset_path', type=str, help='Path to the directory containing the songs')
    parser.add_argument('last_n_songs', type=int, help='Number of songs from the end of the dataset to be used for validation')
    parser.add_argument('output_directory', type=str, help='Directory to save the preprocessed validation data')
    parser.add_argument('segment_length', type=int, help='Length of each segment in seconds')
    parser.add_argument('overlap', type=float, help='Overlap fraction between consecutive segments (e.g. 0.5 for 50 percent overlap)')

    args = parser.parse_args()
    dataset = PreprocessValidationSet(
        data_directory=args.dataset_path,
        sample_rate=44100,
        target_directory=args.output_directory,
        last_n_songs=args.last_n_songs,
        segment_length=args.segment_length,
        overlap=args.overlap
    )
    dataset.preprocess()
