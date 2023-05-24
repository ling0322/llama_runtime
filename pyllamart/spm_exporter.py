import argparse
import sys
import struct
from copy import copy
from functools import partial
from sentencepiece import SentencePieceProcessor
from transformers import GPT2Tokenizer
from typing import Dict, List, Tuple, Callable

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
    def run(cls, tokenize_func: Callable[[str], List[str]], output_file: str):
        """Invoke sentencepiece sample export with arguments"""
        with open(output_file, 'w', encoding="utf-8") as fp:
            for text in cls.SAMPLE_TEXT:
                text_encoded = " ".join(tokenize_func(text))
                fp.write(f"{text}\t{text_encoded}\n")

class Token:
    FLAG_UNK = 1
    FLAG_CONTROL = 2
    FLAG_BYTE = 4
    FLAG_UNUSED = 8

    @classmethod
    def unused(cls, token_id: int) -> 'Token':
        """create an unused token"""
        return Token(token_id, cls.FLAG_UNUSED, b"<UNUSED>", 0)

    def __init__(self, 
                 token_id: int,
                 flag: int = 0,
                 piece: bytes = b"",
                 piece_display: str = "",
                 weight: float = None) -> None:
        self.token_id = token_id
        self.flag = flag
        self.piece = piece
        self.piece_display = piece_display
        self.weight = weight

    def is_unused(self) -> bool:
        return self.FLAG_UNUSED & self.flag != 0

class TokenizerExporter:
    MAGIC_NUMBER = 0x55aa

    @classmethod
    def truncate_display(cls, s: str) -> bytes:
        bs = s.encode('utf-8')
        if len(bs) <= 255:
            return bs
        
        _trunk_repr = lambda s: s.encode('utf-8') + b"...(truncated)"

        bs = _trunk_repr(s)
        while len(bs) >= 256:
            s = s[: -1]
            bs = _trunk_repr(s)

        return bs

    @classmethod
    def write_tokenizer_model(cls, vocab: List[Token], filename: str) -> None:
        """save the sentencepiece exported vocab as llama runtime tokenizer format"""
        with open(filename, 'wb') as fp:
            fp.write(b'LLsp')
            fp.write(struct.pack('<l', len(vocab)))
            fp.write(struct.pack('<h', cls.MAGIC_NUMBER))
            for token in vocab:
                piece_display = cls.truncate_display(token.piece_display)

                fp.write(struct.pack('<b', token.flag))
                fp.write(struct.pack('<B', len(token.piece)))
                fp.write(token.piece)
                fp.write(struct.pack('<B', len(piece_display)))
                fp.write(piece_display)
                fp.write(struct.pack('<f', token.weight))
            fp.write(struct.pack('<h', cls.MAGIC_NUMBER))

    @classmethod
    def write_text_tokenizer_model(cls, vocab: List[Token], filename: str) -> None:
        """save the sentencepiece exported vocab as llama runtime tokenizer format"""
        with open(filename, 'w', encoding="utf-8") as fp:
            for token in vocab:
                piece_display = cls.truncate_display(token.piece_display)
                piece_display = piece_display.decode("utf-8")
                piece = str(token.piece)[2:-1]
                fp.write(f"{token.token_id}\t0x{token.flag:02x}\t{token.weight}\t{piece}\t{piece_display}\n")

class SentencePieceExporter:
    @classmethod
    def read_sentencepiece_model(cls, sp: SentencePieceProcessor) -> List[Token]:
        """read sentencepiece model and return the vocab as list of tuple (flag, token_bytes, weight) and the index
        of the list is token_id
        """
        vocab: List[Token] = []
        for token_id in range(sp.vocab_size()):
            flag = 0
            piece = b''
            piece_display = ""
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
            
            vocab.append(Token(token_id, flag, piece, sp.GetScore(token_id)))
        
        return vocab

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
        cls.write_tokenizer_model(vocab, output_file)

class TransformersBpeExporter(TokenizerExporter):
    """exporter for the BPE tokenizer from transformers"""

    @classmethod
    def to_byte(cls, tokenizer: GPT2Tokenizer, s: str) -> bytes:
        """convert unicode string to bytes according to the char to byte mapping table"""
        b = b''
        byte_decoder = tokenizer.byte_decoder
        for ch in s:
            assert ch in byte_decoder, "invalid character"
            byte_ord = byte_decoder[ch]
            b += byte_ord.to_bytes(1, 'little')
        
        return b
    
    @classmethod
    def read_transformers_bpe_model(cls, tokenizer: GPT2Tokenizer) -> List[Token]:
        """get vocabulary from tokenizer"""
        pieces: Dict[bytes, Token] = {}

        # text and flag
        for piece, piece_id in tokenizer.encoder.items():
            byte_piece = cls.to_byte(tokenizer, piece)
            pieces[piece] = Token(piece_id, flag=0, piece=byte_piece, piece_display=piece)
        
        # weight
        for piece_pair, rank in tokenizer.bpe_ranks.items():
            piece = piece_pair[0] + piece_pair[1]
            if pieces[piece].weight is not None:
                print(f'pair for {piece} already exists')
            pieces[piece].weight = -rank

        # vocab
        vocab: List[Token] = []
        for token_id in range(tokenizer.vocab_size):
            vocab.append(Token.unused(token_id))
        
        for token in pieces.values():
            if not vocab[token.token_id].is_unused():
                raise Exception(f"duplicated token id {token.token_id}")
            vocab[token.token_id] = token

        # special symbols
        for token_id, piece in zip(tokenizer.all_special_ids, tokenizer.all_special_tokens):
            vocab[token_id] = Token(
                token_id,
                Token.FLAG_CONTROL,
                piece=b"",
                piece_display=piece,
                weight=0)
        vocab[tokenizer.unk_token_id].flag |= Token.FLAG_UNK

        # update weight None to 0
        for token in vocab:
            if token.weight is None:
                token.weight = 0

        return vocab

    @classmethod
    def run(cls, argv: List[str]):
        """Export huggingface/transformers BPE tokenizer with arguments"""
        parser = argparse.ArgumentParser(description='export huggingface/transformers BPE model to llm_runtime tokenizer format')
        parser.add_argument('-i', type=str, help='input huggingface/transformers BPE model')
        parser.add_argument('-o', type=str, help='output prefix for fastAlpaca tokenizer')
        args = parser.parse_args(argv)

        output_prefix = args.o

        output_model = f'{output_prefix}.tokenizer.bin'
        output_test_cases = f'{output_prefix}.tokenizer.test_cases.txt'
        
        tokenizer = GPT2Tokenizer.from_pretrained(args.i)
        vocab = cls.read_transformers_bpe_model(tokenizer)
        cls.write_tokenizer_model(vocab, output_model)

        def _tokenize(s: str) -> List[str]:
            return tokenizer.tokenize(s)

        # output test cases
        SentencePieceTestCaseExporter.run(_tokenize, output_test_cases)




if __name__ == '__main__':
    TransformersBpeExporter.run(["-i", "gpt2", "-o", "gpt2_bpe"])
