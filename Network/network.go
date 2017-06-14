package main

import (
	"fmt"
	"hub000.xindong.com/uffy/mnist/MNISTLoader"
	"log"
	"math"
	"math/rand"
	"time"
)

type Network struct {
	LayersNumber int
	Sizes        []int
	Biases       [][]float64
	Weights      [][][][]float64
}

type MNIST struct {
	Data  []float64
	Value int
}

// weights
// [0] first layer
// [1] second net
// ->
// [1] second layer
// [2] third net

func NewNetwork(sizes []int) *Network {
	return &Network{
		LayersNumber: len(sizes),
		Sizes:        sizes,
		Biases:       rand_biases(sizes),
		Weights:      rand_weights(sizes),
	}
}

func rand_biases(sizes []int) [][]float64 {

	biases := make([][]float64, len(sizes)-1)

	for n := range biases {
		biases[n] = make([]float64, sizes[n])

		for m := range biases[n] {
			s1 := rand.NewSource(time.Now().UnixNano())
			r1 := rand.New(s1)
			biases[n][m] = r1.Float64()*2 - 1
		}
	}
	return biases
}

func rand_weights(sizes []int) [][][][]float64 {

	weights := make([][][][]float64, len(sizes)-1)

	for n := range weights {
		if n == len(weights)-1 {
			continue
		}
		weights[n] = make([][][]float64, sizes[n])

		for m := range weights[n] {
			weights[n][m] = make([][]float64, len(sizes)-1)

			for j := range weights[n][m] {
				weights[n][m][j] = make([]float64, sizes[n+1])

				for l := range weights[n][m][j] {
					s1 := rand.NewSource(time.Now().UnixNano())
					r1 := rand.New(s1)
					w := r1.Float64()*2 - 1
					weights[n][m][j][l] = w
				}

			}

		}
	}
	return weights
}

func (net *Network) FeedForward(inputs []float64) []float64 {
	var prev []float64
	var values []float64
	for layer, size := range net.Sizes[1:] {
		if layer == 0 {
			prev = inputs
			continue
		}

		values = make([]float64, size)
		for n := range values {
			var result float64 = 0
			for m := range prev {
				result += net.Weights[layer-1][m][layer][n] * prev[m]
			}
			w := sigmoid(result + net.Biases[layer-1][n])
			values[n] = w
		}
		prev = values
	}

	return prev
}

func (net *Network) SGD(trainingData []MNIST, epochs int, miniBatchSize int, eta float64) {
	n := len(trainingData)

	trainingData = shuffleMNIST(trainingData)

	trainingData = trainingData[0:5000]
	testData := trainingData[5001:6000]

	for j := 0; j < epochs; j++ {
		for i := 0; i < n; i += miniBatchSize {
			end := i + miniBatchSize
			if end > n {
				end = n
			}
			net.updateMiniBatch(trainingData[i:end], eta)
			fmt.Println("total loss: ", net.TotalLoss(trainingData))
		}
	}

	success := 0
	total := 0
	for _, mnist := range testData {
		total++
		a := matrixToInt(net.FeedForward(mnist.Data))

		if a == mnist.Value {
			success++
		}

		fmt.Println("total: ", total, "success", success, "percent: ", success*100/total)
	}
}

func (net *Network) updateMiniBatch(miniBatch []MNIST, eta float64) {

	for i0, wSet0 := range net.Weights {
		for i1, wSet1 := range wSet0 {
			for i2, wSet2 := range wSet1 {
				for i3, w := range wSet2 {
					C0 := net.TotalLoss(miniBatch)

					net.Weights[i0][i1][i2][i3] += eta
					C1 := net.TotalLoss(miniBatch)
					rate := float64(C1-C0) / eta * -eta

					net.Weights[i0][i1][i2][i3] = w + rate
				}
			}
		}
	}

	for b0, bSet0 := range net.Biases {
		for b1, b := range bSet0 {
			C0 := net.TotalLoss(miniBatch)

			net.Biases[b0][b1] += eta
			C1 := net.TotalLoss(miniBatch)
			rate := float64(C1-C0) / eta * -eta

			net.Biases[b0][b1] = b + rate
		}
	}

}
func (net *Network) TotalLoss(batches []MNIST) float64 {
	var total float64 = 0
	for _, batch := range batches {
		value := net.FeedForward(batch.Data)
		L := MSE(batch.matrixValue(), value)
		if math.IsNaN(float64(L)) {
			log.Fatal("L: ", L)
		}
		total += L
	}
	return total
}

func sigmoid(x float64) float64 {
	return 1 / ( 1 + float64(math.Pow(math.E, -float64(x))))
}

func shuffleMNIST(data []MNIST) []MNIST {
	for i := range data {
		r := rand.New(rand.NewSource(time.Now().UnixNano()))
		j := r.Intn(i + 1)
		data[i], data[j] = data[j], data[i]
	}
	return data
}

func MSE(a []float64, b []float64) (cost float64) {
	for n := range a {
		cost += float64(math.Pow(float64(a[n]-b[n]), 2)) / 2
	}
	return
}

func (m *MNIST) matrixValue() []float64 {
	v := make([]float64, 10)
	v[m.Value] = 1

	return v
}

func main() {
	images, labels := MNISTLoader.LoadTrain("/Users/uffy/Workspace/tools/src/hub000.xindong.com/uffy/mnist/data")

	var data []MNIST

	for n, image := range images {
		label := labels[n]

		var inputs []float64 = make([]float64, 784)
		for n, p := range image {
			inputs[n] = float64(p)
		}

		data = append(data, MNIST{
			Data:  inputs,
			Value: int(label),
		})
	}

	net := NewNetwork([]int{784, 30, 10})
	net.SGD(data, 10, 10, 3)
}

func matrixToInt(x []float64) int {
	var max int = 0

	for n, v := range x {
		if v > x[max] {
			max = n
		}
	}

	return max
}

/*

y1 = (x1 w1y1 + b1) + (x2 w2y1 + b1)

y2 = (x1 w1y2 + b2) + (x2 w2y2 + b2)

y3 = (y1 w1y3 + b3) + (y2 w2y3 + b3)

y4 = (y1 w1y4 + b4) + (y2 w2y4 + b4)


Cost:

y3 <-> true result

+

y4 <-> true result


*/
