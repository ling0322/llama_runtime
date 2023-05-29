import argparse
import sys
import struct
import configparser
from copy import copy
from functools import partial
from typing import Dict, List, Tuple, Callable

class TestCaseExporter:
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

class TokenizerConfig:
    def __init__(self,
                 add_prefix_space: bool,
                 split_by_unicode: bool) -> None:
        self.add_prefix_space = add_prefix_space
        self.split_by_unicode = split_by_unicode

    def write_ini(self, filename: str, section: str, model_file: str) -> None:
        config = configparser.ConfigParser()
        config[section] = {}
        config[section]["type"] = "bpe"
        config[section]["model_file"] = model_file
        config[section]["add_prefix_space"] = str(self.add_prefix_space).lower()
        config[section]["split_by_unicode"] = str(self.split_by_unicode).lower()

        with open(filename, 'w', encoding='utf-8') as fp:
            config.write(fp, space_around_delimiters=False)

class SentencePieceModelReader:
    def __init__(self, name: str) -> None:
        from sentencepiece import SentencePieceProcessor
        self._sp = SentencePieceProcessor(model_file=name)

    def to_fastalpaca_model(self) -> List[Token]:
        """convert the sentencepiece model to fastalpaca tokenizer format."""
        vocab: List[Token] = []
        for token_id in range(self._sp.vocab_size()):
            flag = 0
            piece = b''
            piece_display = ""
            if self._sp.IsUnknown(token_id):
                flag = flag | Token.FLAG_UNK
                piece_display = self._sp.IdToPiece(token_id)
            if self._sp.IsControl(token_id):
                flag = flag | Token.FLAG_CONTROL
                piece_display = self._sp.IdToPiece(token_id)
            if self._sp.IsUnused(token_id):
                flag = flag | Token.FLAG_UNUSED
                piece_display = self._sp.IdToPiece(token_id)
            if self._sp.IsByte(token_id):
                flag = flag | Token.FLAG_BYTE
                b = int(self._sp.IdToPiece(token_id)[1: -1], 16)
                b = struct.pack('B', b)
                piece = b
                piece_display = self._sp.IdToPiece(token_id)
            if flag == 0:
                piece = self._sp.IdToPiece(token_id).replace("\u2581", " ").encode('utf-8')
                piece_display = self._sp.IdToPiece(token_id)

            piece = piece
            piece_display = piece_display
            
            vocab.append(Token(token_id, flag, piece, piece_display, self._sp.GetScore(token_id)))
        
        return vocab
    
    def encode_as_pieces(self, s: str) -> List[str]:
        return self._sp.EncodeAsPieces(s)

    def tokenizer_config(self) -> TokenizerConfig:
        return TokenizerConfig(
            add_prefix_space=True,
            split_by_unicode=True)

class TransformersBpeModelReader:
    """model reader for the BPE tokenizer from transformers"""

    def __init__(self, name: str) -> None:
        from transformers import GPT2Tokenizer
        self._tokenizer = GPT2Tokenizer.from_pretrained(name)

    def _to_byte(self, s: str) -> bytes:
        """convert unicode string to bytes according to the char to byte mapping table"""
        b = b''
        byte_decoder = self._tokenizer.byte_decoder
        for ch in s:
            assert ch in byte_decoder, "invalid character"
            byte_ord = byte_decoder[ch]
            b += byte_ord.to_bytes(1, 'little')
        
        return b

    def to_fastalpaca_model(self) -> List[Token]:
        """convert the BPE model to fastalpaca tokenizer format."""
        pieces: Dict[bytes, Token] = {}

        # text and flag
        for piece, piece_id in self._tokenizer.encoder.items():
            byte_piece = self._to_byte(piece)
            pieces[piece] = Token(piece_id, flag=0, piece=byte_piece, piece_display=piece)
        
        # weight
        for piece_pair, rank in self._tokenizer.bpe_ranks.items():
            piece = piece_pair[0] + piece_pair[1]
            if pieces[piece].weight is not None:
                print(f'pair for {piece} already exists')
            pieces[piece].weight = -rank

        # vocab
        vocab: List[Token] = []
        for token_id in range(self._tokenizer.vocab_size):
            vocab.append(Token.unused(token_id))
        
        for token in pieces.values():
            if not vocab[token.token_id].is_unused():
                raise Exception(f"duplicated token id {token.token_id}")
            vocab[token.token_id] = token

        # special symbols
        for token_id, piece in zip(self._tokenizer.all_special_ids, self._tokenizer.all_special_tokens):
            vocab[token_id] = Token(
                token_id,
                Token.FLAG_CONTROL,
                piece=b"",
                piece_display=piece,
                weight=0)
        vocab[self._tokenizer.unk_token_id].flag |= Token.FLAG_UNK

        # update weight None to 0
        for token in vocab:
            if token.weight is None:
                token.weight = 0

        return vocab
    
    def encode_as_pieces(self, s: str) -> List[str]:
        return self._tokenizer.tokenize(s)

    def tokenizer_config(self) -> TokenizerConfig:
        return TokenizerConfig(
            add_prefix_space=False,
            split_by_unicode=False)

class Util:
    @classmethod
    def escape_string(cls, s):
        e = ""
        for ch in s:
            if ord(ch) <= 32:
                ch = f"\\x{ord(ch):02x}"
            e += ch
        return e

    @classmethod
    def escape_bytes(cls, s):
        e = b""
        for ch in s:
            if ch <= 32 or ch >= 127:
                ch = f"\\x{ch:02x}".encode("utf-8")
            else:
                ch = ch.to_bytes(1, 'little')
            e += ch
        return e

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

class TokenizerExporter:
    MAGIC_NUMBER = 0x55aa

    @classmethod
    def write_tokenizer_model(cls, vocab: List[Token], filename: str) -> None:
        """save the sentencepiece exported vocab as llama runtime tokenizer format"""
        with open(filename, 'wb') as fp:
            fp.write(b'LLsp')
            fp.write(struct.pack('<l', len(vocab)))
            fp.write(struct.pack('<h', cls.MAGIC_NUMBER))
            for token in vocab:
                piece_display = Util.truncate_display(token.piece_display)

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
                piece = token.piece.decode()
                fp.write(f"{token.token_id}\t0x{token.flag:02x}\t{token.weight}\t{piece}\t{piece_display}\n")


    @classmethod
    def run(cls, argv: List[str]):
        """Export huggingface/transformers BPE tokenizer with arguments"""
        parser = argparse.ArgumentParser(description='export SentencePiece or huggingface-transformers BPE model to fastalpaca tokenizer format.')
        parser.add_argument('-t', type=str, help='tokenizer model type. "spm" for SentencePiece BPE model, "transformers" for huggingface/transformers BPE model.')
        parser.add_argument('-i', type=str, help='tokenizer model name or path.')
        parser.add_argument('-o', type=str, help='output prefix for fastalpaca tokenizer.')
        args = parser.parse_args(argv)

        model_type = args.t
        model_name = args.i
        if not model_name:
            print("ERROR: invalid model path or name.")
            parser.print_usage()
            sys.exit(1)

        model = None
        if model_type == "spm":
            model = SentencePieceModelReader(model_name)
        elif model_type == "transformers":
            model = TransformersBpeModelReader(model_name)
        else:
            print("ERROR: invalid model type.")
            parser.print_usage()
            sys.exit(1)


        output_prefix = args.o

        output_model = f'{output_prefix}.tokenizer.bin'
        output_ini = f'{output_prefix}.tokenizer.ini'
        output_test_cases = f'{output_prefix}.tokenizer.test_cases.txt'
        
        cls.write_tokenizer_model(model.to_fastalpaca_model(), output_model)
        TestCaseExporter.run(model.encode_as_pieces, output_test_cases)
        model.tokenizer_config().write_ini(output_ini, "tokenizer", output_model)

if __name__ == '__main__':
    #TokenizerExporter.run(["-t", "spm", "-i", "pyllamart/tokenizer.model", "-o", "llama_spm"])
    TokenizerExporter.run(["-t", "transformers", "-i", "gpt2", "-o", "gpt2_bpe"])
