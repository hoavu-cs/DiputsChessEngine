package main

import (
	"fmt"
	"math/bits"
	"math/rand"
)

func file(sq int) int {
	return sq % 8
}

func rank(sq int) int {
	return sq / 8
}

// attack mask for bishops
func bishop_mask(sq int) uint64 {
	r := rank(sq)
	c := file(sq)
	var bb uint64
	for i, j := r+1, c+1; i < 7 && j < 7; i, j = i+1, j+1 {
		bb |= uint64(1) << (i*8 + j)
	}
	for i, j := r-1, c-1; i > 0 && j > 0; i, j = i-1, j-1 {
		bb |= uint64(1) << (i*8 + j)
	}
	for i, j := r+1, c-1; i < 7 && j > 0; i, j = i+1, j-1 {
		bb |= uint64(1) << (i*8 + j)
	}
	for i, j := r-1, c+1; i > 0 && j < 7; i, j = i-1, j+1 {
		bb |= uint64(1) << (i*8 + j)
	}
	return bb
}

// attack mask for the bishop given the square and blocker(s)
func bishop_attacks_for(sq int, blockers uint64) uint64 {
	r := rank(sq)
	c := file(sq)
	var bb uint64
	for i, j := r+1, c+1; i <= 7 && j <= 7; i, j = i+1, j+1 {
		bb |= uint64(1) << (i*8 + j)
		if blockers&(uint64(1)<<(i*8+j)) != 0 {
			break
		}
	}
	for i, j := r-1, c-1; i >= 0 && j >= 0; i, j = i-1, j-1 {
		bb |= uint64(1) << (i*8 + j)
		if blockers&(uint64(1)<<(i*8+j)) != 0 {
			break
		}
	}
	for i, j := r+1, c-1; i <= 7 && j >= 0; i, j = i+1, j-1 {
		bb |= uint64(1) << (i*8 + j)
		if blockers&(uint64(1)<<(i*8+j)) != 0 {
			break
		}
	}
	for i, j := r-1, c+1; i >= 0 && j <= 7; i, j = i-1, j+1 {
		bb |= uint64(1) << (i*8 + j)
		if blockers&(uint64(1)<<(i*8+j)) != 0 {
			break
		}
	}
	return bb
}

// attack mask for rooks
func rook_attacks_for(sq int, blockers uint64) uint64 {
	r := rank(sq)
	c := file(sq)
	var bb uint64
	for i := r + 1; i <= 7; i++ {
		bb |= uint64(1) << (i*8 + c)
		if blockers&(uint64(1)<<(i*8+c)) != 0 {
			break
		}
	}
	for i := r - 1; i >= 0; i-- {
		bb |= uint64(1) << (i*8 + c)
		if blockers&(uint64(1)<<(i*8+c)) != 0 {
			break
		}
	}
	for j := c + 1; j <= 7; j++ {
		bb |= uint64(1) << (r*8 + j)
		if blockers&(uint64(1)<<(r*8+j)) != 0 {
			break
		}
	}
	for j := c - 1; j >= 0; j-- {
		bb |= uint64(1) << (r*8 + j)
		if blockers&(uint64(1)<<(r*8+j)) != 0 {
			break
		}
	}
	return bb
}

// attack mask for the rooks given the square and blocker(s)
func rook_mask(sq int) uint64 {
	r := rank(sq)
	c := file(sq)
	var bb uint64
	for i := r + 1; i < 7; i++ {
		bb |= uint64(1) << (i*8 + c)
	}
	for i := r - 1; i > 0; i-- {
		bb |= uint64(1) << (i*8 + c)
	}
	for j := c + 1; j < 7; j++ {
		bb |= uint64(1) << (r*8 + j)
	}
	for j := c - 1; j > 0; j-- {
		bb |= uint64(1) << (r*8 + j)
	}
	return bb
}

// Return the bitboards of attacked configurations of the bishop on a given square
func bishop_attacks() []map[uint64]uint64 {
	attacks := make([]map[uint64]uint64, 64)
	for sq := range 64 {
		mask := bishop_mask(sq)
		attacks[sq] = make(map[uint64]uint64)
		// iterate through all possible configurations of blockers
		for blockers := mask; blockers != 0; blockers = (blockers - 1) & mask {
			attacks[sq][blockers] = bishop_attacks_for(sq, blockers)
		}
		attacks[sq][0] = bishop_attacks_for(sq, 0)
	}
	return attacks
}

// Return the bitboards of attacked configurations of the rook on a given square
func rook_attacks() []map[uint64]uint64 {
	attacks := make([]map[uint64]uint64, 64)
	for sq := range 64 {
		mask := rook_mask(sq)
		attacks[sq] = make(map[uint64]uint64)
		for blockers := mask; blockers != 0; blockers = (blockers - 1) & mask {
			attacks[sq][blockers] = rook_attacks_for(sq, blockers)
		}
		attacks[sq][0] = rook_attacks_for(sq, 0)
	}
	return attacks
}

// Search magic numbers
func search_magic_bishop(sq int) uint64 {
	mask := bishop_mask(sq)
	n_bits := bits.OnesCount64(mask)
	attacks := bishop_attacks()[sq]
	var magic uint64

	for {
		magic = rand.Uint64() & rand.Uint64() & rand.Uint64()
		table := make([]uint64, 1<<n_bits)
		used := make([]bool, 1<<n_bits)
		fail := false

		for blockers, attacks := range attacks {
			idx := (blockers * magic) >> (64 - n_bits)
			if !used[idx] {
				used[idx] = true
				table[idx] = attacks
			} else if table[idx] != attacks {
				fail = true
				break
			}
		}

		if !fail {
			break
		}
	}
	return magic
}

func search_magic_rook(sq int) uint64 {
	mask := rook_mask(sq)
	n_bits := bits.OnesCount64(mask)
	attacks := rook_attacks()[sq]
	var magic uint64

	for {
		magic = rand.Uint64() & rand.Uint64() & rand.Uint64()
		table := make([]uint64, 1<<n_bits)
		used := make([]bool, 1<<n_bits)
		fail := false

		for blockers, atk := range attacks {
			idx := (blockers * magic) >> (64 - n_bits)
			if !used[idx] {
				used[idx] = true
				table[idx] = atk
			} else if table[idx] != atk {
				fail = true
				break
			}
		}

		if !fail {
			break
		}
	}
	return magic
}

func main() {
	bishop_magics := make([]uint64, 64)
	rook_magics := make([]uint64, 64)
	for sq := 0; sq < 64; sq++ {
		bishop_magics[sq] = search_magic_bishop(sq)
		rook_magics[sq] = search_magic_rook(sq)
	}

	fmt.Println("const BISHOP_MAGIC = UInt64[")
	for _, m := range bishop_magics {
		fmt.Printf("\t0x%X,\n", m)
	}
	fmt.Println("]")
	fmt.Println("const ROOK_MAGIC = UInt64[")
	for _, m := range rook_magics {
		fmt.Printf("\t0x%X,\n", m)
	}
	fmt.Println("]")

}
