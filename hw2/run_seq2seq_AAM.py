# Import Libraries
import argparse
import json
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from bleu_eval import BLEU

# Vocabulary class
class Vocabulary:
    def __init__(self, min_word_count=3):
        self.word2index = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.index2word = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.word_count = {}
        self.min_word_count = min_word_count

    def build_vocab(self, captions):
        word_counter = {}
        for caption in captions:
            tokens = caption.lower().split()
            for word in tokens:
                word_counter[word] = word_counter.get(word, 0) + 1

        for word, count in word_counter.items():
            if count >= self.min_word_count:
                index = len(self.word2index)
                self.word2index[word] = index
                self.index2word[index] = word

    def encode_sentence(self, sentence):
        tokens = sentence.lower().split()
        return [self.word2index.get(word, self.word2index["<UNK>"]) for word in tokens]

    def decode_sentence(self, indices):
        return [self.index2word.get(idx, "<UNK>") for idx in indices]

# Dataset class
class VideoCaptionDataset(Dataset):
    def __init__(self, video_dir, caption_file, vocab):
        self.video_dir = video_dir
        self.vocab = vocab

        # Load captions
        with open(caption_file, 'r') as f:
            self.captions = json.load(f)

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        caption_data = self.captions[idx]
        video_id = caption_data['id']
        caption = caption_data['caption'][0]  # First caption as reference

        # Load video features
        video_features = np.load(os.path.join(self.video_dir, f"{video_id}.npy"))

        # Encode captions into indices
        encoded_caption = self.vocab.encode_sentence(caption)

        # Convert to tensors
        video_features = torch.tensor(video_features, dtype=torch.float32)
        encoded_caption = torch.tensor(encoded_caption, dtype=torch.long)

        return video_features, encoded_caption

# Encoder with LSTM
class EncoderLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.3):
        super(EncoderLSTM, self).__init__()
        # Set dropout to 0 if num_layers is 1 (since dropout has no effect for 1 layer)
        dropout = 0 if num_layers == 1 else dropout
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)

    def forward(self, video_features):
        outputs, (hidden, cell) = self.lstm(video_features)
        return outputs, hidden, cell

# Attention mechanism
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_size * 2, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        # Check if in beam search (batch size = 1) and adjust encoder outputs accordingly
        if hidden.size(0) == 1 and encoder_outputs.size(0) != 1:
            encoder_outputs = encoder_outputs[:1, :, :]  # Slice encoder outputs to keep batch size 1

        batch_size, seq_len, _ = encoder_outputs.size()

        # Repeat hidden state across the sequence length
        hidden = hidden.unsqueeze(1).repeat(1, seq_len, 1)

        # Concatenate hidden state with encoder outputs
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))

        # Calculate attention weights
        attention = self.v(energy).squeeze(2)
        return torch.softmax(attention, dim=1)

    def apply_attention(self, attention_weights, encoder_outputs):
        # Check if in beam search (batch size = 1) and adjust encoder outputs accordingly
        if attention_weights.size(0) == 1 and encoder_outputs.size(0) != 1:
            encoder_outputs = encoder_outputs[:1, :, :]  

        # Perform batch matrix multiplication to get context
        return torch.bmm(attention_weights.unsqueeze(1), encoder_outputs).squeeze(1)

# Decoder with LSTM and Attention
class DecoderLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers=1, dropout=0.3):
        super(DecoderLSTM, self).__init__()
        dropout = 0 if num_layers == 1 else dropout
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(hidden_size + embedding_dim, hidden_size, num_layers, dropout=dropout, batch_first=True)
        self.fc_out = nn.Linear(hidden_size, vocab_size)
        self.attention = Attention(hidden_size)

    def forward(self, input_word, hidden, cell, encoder_outputs):
        embedded = self.embedding(input_word.unsqueeze(1))  

        attention_weights = self.attention(hidden[-1], encoder_outputs)  
        context = self.attention.apply_attention(attention_weights, encoder_outputs) 

        # Ensure that embedded and context match in terms of batch size during beam search
        if embedded.size(0) == 1 and context.size(0) != 1:  # If batch size is 1 (beam search), adjust context
            context = context[:1]  # Keep the first element to match the batch size of 1

        embedded = embedded.squeeze(1)  
        lstm_input = torch.cat((embedded, context), dim=1).unsqueeze(1)  
        output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))  
        predictions = self.fc_out(output.squeeze(1))  
        return predictions, hidden, cell

# Beam Search for Decoding
def beam_search(decoder, encoder_outputs, hidden, cell, vocab, beam_width=3, max_len=28):
    decoder_input = torch.tensor([vocab.word2index["<SOS>"]]).cuda()

    # Adjust hidden and cell to match the batch size of beam search (1)
    hidden = hidden[:, :1, :]
    cell = cell[:, :1, :]

    beam = [(0, [decoder_input], hidden, cell)]

    for _ in range(max_len):
        new_beam = []
        for log_prob, sentence, hidden, cell in beam:
            decoder_input = sentence[-1]
            if decoder_input == vocab.word2index["<EOS>"]:
                new_beam.append((log_prob, sentence, hidden, cell))
                continue

            predictions, hidden, cell = decoder(decoder_input, hidden, cell, encoder_outputs)
            predictions = torch.log_softmax(predictions, dim=1)
            topk_probs, topk_indices = torch.topk(predictions, beam_width)

            for i in range(beam_width):
                next_token = topk_indices[0][i].unsqueeze(0)
                new_log_prob = log_prob + topk_probs[0][i].item()
                new_sentence = sentence + [next_token]
                new_beam.append((new_log_prob, new_sentence, hidden, cell))

        beam = sorted(new_beam, key=lambda x: x[0], reverse=True)[:beam_width]

    best_sentence = beam[0][1]
    return [token.item() for token in best_sentence]

def evaluate_bleu_score(encoder, decoder, dataloader, vocab, beam_width=3):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  

    encoder.eval()
    decoder.eval()

    all_bleu_scores = []
    for video_features, captions in dataloader:
        
        video_features = video_features.to(device)
        captions = captions.to(device)

        encoder_outputs, hidden, cell = encoder(video_features)
        predicted_sentence = beam_search(decoder, encoder_outputs, hidden, cell, vocab, beam_width)

        # Convert the list of predicted tokens to a string
        decoded_prediction = ' '.join(vocab.decode_sentence(predicted_sentence))
        decoded_reference = ' '.join(vocab.decode_sentence(captions[0].tolist()))

        # Calculate BLEU score
        bleu_score = BLEU(decoded_prediction, decoded_reference, flag=False)
        all_bleu_scores.append(bleu_score)

    return sum(all_bleu_scores) / len(all_bleu_scores)

# Training Function
def train_model(encoder, decoder, dataloader, vocab, epochs=200, learning_rate=0.001, teacher_forcing_ratio=0.5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()

    best_bleu_score = 0.0

    if not os.path.exists('Best Model'):
        os.makedirs('Best Model')

    for epoch in range(epochs):
        encoder.train()
        decoder.train()

        total_loss = 0
        for video_features, captions in dataloader:
            video_features = video_features.to(device)
            captions = captions.to(device)

            optimizer.zero_grad()
            encoder_outputs, hidden, cell = encoder(video_features)

            decoder_input = torch.tensor([vocab.word2index["<SOS>"]] * video_features.size(0)).to(device)
            loss = 0

            for t in range(1, captions.size(1)):
                predictions, hidden, cell = decoder(decoder_input, hidden, cell, encoder_outputs)
                loss += criterion(predictions, captions[:, t])

                teacher_force = torch.rand(1).item() < teacher_forcing_ratio
                decoder_input = captions[:, t] if teacher_force else predictions.argmax(1)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}, Average Loss: {avg_loss:.4f}")

        # Evaluate BLEU score
        bleu_score = evaluate_bleu_score(encoder, decoder, dataloader, vocab, beam_width=3)
        print(f"Epoch {epoch+1}, Average BLEU Score: {bleu_score:.4f}")

        # Save model if BLEU score improves
        if bleu_score > best_bleu_score:
            best_bleu_score = bleu_score
            print(f"New best BLEU score: {best_bleu_score:.4f}. Saving model...")

            torch.save(encoder.state_dict(), os.path.join('Best Model', 'best_encoder.pth'))
            torch.save(decoder.state_dict(), os.path.join('Best Model', 'best_decoder.pth'))


# Function to Generate Captions for All Test Videos and Calculate BLEU
def evaluate_and_save_results(encoder, decoder, dataloader, vocab, output_file, beam_width=3):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    encoder.eval()
    decoder.eval()

    with open(output_file, 'w') as f:
        for batch_idx, (video_features, captions) in enumerate(dataloader):
            video_features = video_features.to(device)

            encoder_outputs, hidden, cell = encoder(video_features)

            for idx in range(video_features.size(0)):
                predicted_sentence = beam_search(
                    decoder, encoder_outputs[idx:idx+1], hidden[:, idx:idx+1, :], cell[:, idx:idx+1, :], vocab, beam_width=3
                )
                decoded_prediction = vocab.decode_sentence(predicted_sentence)

                # Filter out special tokens
                filtered_prediction = [word for word in decoded_prediction if word not in ['<UNK>', '<PAD>', '<SOS>', '<EOS>']]
                decoded_prediction_text = ' '.join(filtered_prediction)

                video_id = dataloader.dataset.captions[batch_idx * dataloader.batch_size + idx]['id']
                f.write(f"{video_id},{decoded_prediction_text}\n")

    print(f"Results saved to {output_file}")


# Custom collate function to pad captions to the same length
def collate_fn(batch):
    video_features, captions = zip(*batch)
    video_features = torch.stack(video_features)
    captions = [caption.clone().detach() for caption in captions]
    captions_padded = pad_sequence(captions, batch_first=True, padding_value=0)
    return video_features, captions_padded


# Main function to parse arguments and run the training/evaluation
def main():
    parser = argparse.ArgumentParser(description='Video Captioning with Seq2Seq and Attention')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to the data directory')
    parser.add_argument('--output', type=str, required=True, help='Output file to save test results')
    args = parser.parse_args()

    # Paths to training and testing data
    train_video_dir = os.path.join(args.data_dir, 'training_data', 'feat')
    train_caption_file = os.path.join(args.data_dir, 'training_label.json')
    test_video_dir = os.path.join(args.data_dir, 'testing_data', 'feat')
    test_caption_file = os.path.join(args.data_dir, 'testing_label.json')

    # Load Vocabulary
    vocab = Vocabulary()
    with open(train_caption_file, 'r') as f:
        captions = [item['caption'][0] for item in json.load(f)]
    vocab.build_vocab(captions)

    # Dataloaders
    train_dataset = VideoCaptionDataset(train_video_dir, train_caption_file, vocab)
    test_dataset = VideoCaptionDataset(test_video_dir, test_caption_file, vocab)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

    # Model parameters
    input_size = 4096
    hidden_size = 512
    vocab_size = len(vocab.word2index)
    embedding_dim = 300

    # Initialize encoder and decoder
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder = EncoderLSTM(input_size, hidden_size).to(device)
    decoder = DecoderLSTM(vocab_size, embedding_dim, hidden_size).to(device)

    # Train model
    train_model(encoder, decoder, train_loader, vocab)

    # Evaluate and save results for all test videos
    evaluate_and_save_results(encoder, decoder, test_loader, vocab, args.output, beam_width=3)


if __name__ == "__main__":
    main()
