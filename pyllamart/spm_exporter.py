import argparse
import sys
import struct
from sentencepiece import SentencePieceProcessor
from typing import List, Tuple

class SentencePieceExporter:
    FLAG_UNK = 1
    FLAG_CONTROL = 2
    FLAG_UNUSED = 4

    MAGIC_NUMBER = 0x55aa

    @classmethod
    def read_sentencepiece_model(cls, sp: SentencePieceProcessor) -> List[Tuple[int, bytes, float]]:
        """read sentencepiece model and return the vocab as list of tuple (flag, token_bytes, weight) and the index
        of the list is token_id
        """
        vocab: List[Tuple[int, bytes, float]] = []
        for token_id in range(sp.vocab_size()):
            flag = 0
            token_string = b''
            if sp.IsUnknown(token_id):
                flag = flag & cls.FLAG_UNK
            if sp.IsControl(token_id):
                flag = flag & cls.FLAG_CONTROL
            if sp.IsUnused(token_id):
                flag = flag & cls.FLAG_UNUSED

            if sp.IsByte(token_id):
                b = int(sp.IdToPiece(token_id)[1: -1], 16)
                b = struct.pack('B', b)
                token_string = b
            elif flag == 0:
                token_string = sp.IdToPiece(token_id).encode('utf-8')
            
            vocab.append((flag, token_string, sp.GetScore(token_id)))
        
        return vocab

    @classmethod
    def save_llamart_tokenizer(cls, vocab: List[Tuple[int, bytes, float]], filename: str) -> None:
        """save the sentencepiece exported vocab as llama runtime tokenizer format"""
        with open(filename, 'wb') as fp:
            fp.write(b'LLsp')
            fp.write(struct.pack('<l', len(vocab)))
            fp.write(struct.pack('<h', cls.MAGIC_NUMBER))
            for flag, token_string, weight in vocab:
                fp.write(struct.pack('<b', flag))
                fp.write(struct.pack('<b', len(token_string)))
                fp.write(token_string)
                fp.write(struct.pack('<f', weight))
            fp.write(struct.pack('<h', cls.MAGIC_NUMBER))

    @classmethod
    def run(cls, argv: List[str]):
        """Invoke whisper ONNX export with arguments"""
        parser = argparse.ArgumentParser(description='export SentencePiece model to llama_runtime tokenizer format')
        parser.add_argument('-i', type=str, help='input SentencePiece model')
        parser.add_argument('-o', type=str, help='output llama_runtime tokenizer')
        args = parser.parse_args(argv)

        model_file = args.i
        output_file = args.o
        
        sp = SentencePieceProcessor(model_file=model_file)
        vocab = cls.read_sentencepiece_model(sp)
        cls.save_llamart_tokenizer(vocab, output_file)


if __name__ == '__main__':
    SentencePieceExporter.run(sys.argv[1: ])
