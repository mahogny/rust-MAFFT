/// An alphabet mapping between characters and internal indices.
pub trait Alphabet {
    fn chars(&self) -> &[u8];
    fn len(&self) -> usize {
        self.chars().len()
    }
    fn char_to_index(&self, ch: u8) -> Option<usize>;
    fn index_to_char(&self, idx: usize) -> Option<u8>;

    /// Build a 256-element lookup table: ASCII char -> internal index.
    /// Unmapped characters get 0xFF.
    fn build_amino_map(&self) -> [u8; 256] {
        let mut map = [0xFFu8; 256];
        for (i, &ch) in self.chars().iter().enumerate() {
            map[ch as usize] = i as u8;
        }
        map
    }
}

/// Protein alphabet: ARNDCQEGHILKMFPSTWYVBZX.-J (26 characters).
///
/// Indices 0-19: standard amino acids.
/// 20=B (Asx), 21=Z (Glx), 22=X (unknown), 23='.', 24='-', 25=J.
pub struct ProteinAlphabet {
    pub chars: [u8; 26],
    pub groups: [u8; 26],
}

pub static PROTEIN_ALPHABET: ProteinAlphabet = ProteinAlphabet {
    chars: *b"ARNDCQEGHILKMFPSTWYVBZX.-J",
    groups: [
        0, 3, 2, 2, 5, 2, 2, 0, 3, 1, 1, 3, 1, 4, 0, 0, 0, 4, 4, 1, 2, 2, 6, 6, 6, 1,
    ],
};

impl Alphabet for ProteinAlphabet {
    fn chars(&self) -> &[u8] {
        &self.chars
    }
    fn len(&self) -> usize {
        26
    }

    fn char_to_index(&self, ch: u8) -> Option<usize> {
        self.chars.iter().position(|&c| c == ch)
    }

    fn index_to_char(&self, idx: usize) -> Option<u8> {
        self.chars.get(idx).copied()
    }
}

/// DNA alphabet: agctuAGCTUnNbdhkmnrsvwyx-O (26 characters).
///
/// Indices 0-4: a,g,c,t,u (lowercase).
/// Indices 5-9: A,G,C,T,U (uppercase).
/// 10-11: n,N. 12-22: ambiguity codes. 23=x, 24='-', 25=O.
pub struct DnaAlphabet {
    pub chars: [u8; 26],
    pub groups: [u8; 26],
}

pub static DNA_ALPHABET: DnaAlphabet = DnaAlphabet {
    chars: *b"agctuAGCTUnNbdhkmnrsvwyx-O",
    groups: [
        0, 1, 2, 3, 3, 0, 1, 2, 3, 3, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
    ],
};

impl Alphabet for DnaAlphabet {
    fn chars(&self) -> &[u8] {
        &self.chars
    }
    fn len(&self) -> usize {
        26
    }

    fn char_to_index(&self, ch: u8) -> Option<usize> {
        self.chars.iter().position(|&c| c == ch)
    }

    fn index_to_char(&self, idx: usize) -> Option<u8> {
        self.chars.get(idx).copied()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn protein_alphabet_mapping() {
        assert_eq!(PROTEIN_ALPHABET.char_to_index(b'A'), Some(0));
        assert_eq!(PROTEIN_ALPHABET.char_to_index(b'R'), Some(1));
        assert_eq!(PROTEIN_ALPHABET.char_to_index(b'V'), Some(19));
        assert_eq!(PROTEIN_ALPHABET.char_to_index(b'-'), Some(24));
        assert_eq!(PROTEIN_ALPHABET.char_to_index(b'Z'), Some(21)); // Glx ambiguity
    }

    #[test]
    fn dna_alphabet_mapping() {
        assert_eq!(DNA_ALPHABET.char_to_index(b'a'), Some(0));
        assert_eq!(DNA_ALPHABET.char_to_index(b'g'), Some(1));
        assert_eq!(DNA_ALPHABET.char_to_index(b'A'), Some(5));
        assert_eq!(DNA_ALPHABET.char_to_index(b'-'), Some(24));
    }

    #[test]
    fn amino_map_roundtrip() {
        let map = PROTEIN_ALPHABET.build_amino_map();
        assert_eq!(map[b'A' as usize], 0);
        assert_eq!(map[b'R' as usize], 1);
        assert_eq!(map[b'-' as usize], 24);
        assert_eq!(map[b'z' as usize], 0xFF); // lowercase z not in alphabet
    }
}
