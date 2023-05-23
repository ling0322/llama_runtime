import argparse
import sys
import struct
from sentencepiece import SentencePieceProcessor
from transformers import GPT2Tokenizer
from typing import List, Tuple

class SentencePieceTestCaseExporter:
    SAMPLE_TEXT = [
        "The Touhou Project (Japanese: 東方Project, Hepburn: Tōhō Purojekuto), ",
        "also known simply as Touhou (東方, literally ",
        '''"Eastern" or "Oriental"), is a bullet hell shoot 'em up video game series created by one-man ''',
        "independent Japanese",
        "doujin soft developer Team Shanghai Alice.",
        """Since 1995,[1][2] the team's member, Jun'ya "ZUN" Ōta, has independently""",
        "developed programming, graphics, writing, and music for the series, self-publishing 18 mainline games and six",
        "spin-offs as of August 2022. ",
        "ZUN has also produced related print works and music albums, and collaborated with",
        "developer Twilight Frontier on seven official Touhou spin-offs, most being fighting games.[3]",
        "🐱"
    ]

    @classmethod
    def run(cls, argv: List[str]):
        """Invoke sentencepiece sample export with arguments"""
        parser = argparse.ArgumentParser(description='export SentencePiece model to llama_runtime tokenizer format')
        parser.add_argument('-m', type=str, help='input SentencePiece model')
        parser.add_argument('-o', type=str, help='output test case file')
        args = parser.parse_args(argv)

        model_file = args.m
        output_file = args.o
        
        sp = SentencePieceProcessor(model_file=model_file)
        with open(output_file, 'w', encoding='utf-8') as fp:
            for text in cls.SAMPLE_TEXT:
                text_encoded = ' '.join(sp.EncodeAsPieces(text))
                fp.write(f'{text}\t{text_encoded}\n')

class SentencePieceExporter:
    FLAG_UNK = 1
    FLAG_CONTROL = 2
    FLAG_BYTE = 4
    FLAG_UNUSED = 8
    MAGIC_NUMBER = 0x55aa

    @classmethod
    def read_sentencepiece_model(cls, sp: SentencePieceProcessor) -> List[Tuple[int, bytes, float]]:
        """read sentencepiece model and return the vocab as list of tuple (flag, token_bytes, weight) and the index
        of the list is token_id
        """
        vocab: List[Tuple[int, bytes, float]] = []
        for token_id in range(sp.vocab_size()):
            flag = 0
            token_bytes = b''
            if sp.IsUnknown(token_id):
                flag = flag | cls.FLAG_UNK
            if sp.IsControl(token_id):
                flag = flag | cls.FLAG_CONTROL
            if sp.IsUnused(token_id):
                flag = flag | cls.FLAG_UNUSED
            if sp.IsByte(token_id):
                flag = flag | cls.FLAG_BYTE
                b = int(sp.IdToPiece(token_id)[1: -1], 16)
                b = struct.pack('B', b)
                token_bytes = b
            if flag == 0:
                token_bytes = sp.IdToPiece(token_id).encode('utf-8')
            
            vocab.append((flag, token_bytes, sp.GetScore(token_id)))
        
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
        """Invoke sentencepipce BPE model export with arguments"""
        parser = argparse.ArgumentParser(description='export SentencePiece model to llama_runtime tokenizer format')
        parser.add_argument('-i', type=str, help='input SentencePiece model')
        parser.add_argument('-o', type=str, help='output llama_runtime tokenizer')
        args = parser.parse_args(argv)

        model_file = args.i
        output_file = args.o
        
        sp = SentencePieceProcessor(model_file=model_file)
        vocab = cls.read_sentencepiece_model(sp)
        cls.save_llamart_tokenizer(vocab, output_file)

class BpeExporter:
    """exporter for the BPE tokenizer from transformers"""
    def __init__(self, tokenizer: GPT2Tokenizer) -> None:
        self._byte_decoder = tokenizer.byte_decoder
        self._tokenizer = tokenizer

    def _to_byte(self, s: str) -> bytes:
        """convert unicode string to bytes according to the char to byte mapping table"""
        b = b''
        for ch in s:
            assert ch in self._byte_decoder, "invalid character"
            byte_ord = self._byte_decoder[ch]
            b += byte_ord.to_bytes(length=1)
        
        return b


    def read_bpe_model(cls, sp: SentencePieceProcessor) -> List[Tuple[int, bytes, float]]:
        """read sentencepiece model and return the vocab as list of tuple (flag, token_bytes, weight) and the index
        of the list is token_id
        """
        vocab: List[Tuple[int, bytes, float]] = []
        for token_id in range(sp.vocab_size()):
            flag = 0
            token_bytes = b''
            if sp.IsUnknown(token_id):
                flag = flag | cls.FLAG_UNK
            if sp.IsControl(token_id):
                flag = flag | cls.FLAG_CONTROL
            if sp.IsUnused(token_id):
                flag = flag | cls.FLAG_UNUSED
            if sp.IsByte(token_id):
                flag = flag | cls.FLAG_BYTE
                b = int(sp.IdToPiece(token_id)[1: -1], 16)
                b = struct.pack('B', b)
                token_bytes = b
            if flag == 0:
                token_bytes = sp.IdToPiece(token_id).encode('utf-8')
            
            vocab.append((flag, token_bytes, sp.GetScore(token_id)))
        
        return vocab

if __name__ == '__main__':
    
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    print(tokenizer("token\nizer 22\n "))

    sys.exit(2)

    SentencePieceExporter.run(sys.argv[1: ])
