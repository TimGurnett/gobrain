package gobrain

import (
	"math"
	"math/rand"
)

func random(a, b float64) float64 {
	return (b-a)*rand.Float64() + a
}

func matrix(I, J int) [][]float64 {
	m := make([][]float64, I)
	for i := 0; i < I; i++ {
		m[i] = make([]float64, J)
	}
	return m
}

func vector(I int, fill float64) []float64 {
	v := make([]float64, I)
	for i := 0; i < I; i++ {
		v[i] = fill
	}
	return v
}

func leakyReLu(x float64) float64 {
	if x < 0 {
		return x * 0.01
	}
	return x
}

func dleakyReLu(x float64) float64 {
	if x > 0 {
		return 1
	}
	return 0.01
}

func sigmoid(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}

func dsigmoid(y float64) float64 {
	return y * (1 - y)
}
