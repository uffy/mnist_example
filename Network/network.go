package main

import (
	"fmt"
	"log"
	"math"
	"math/rand"
	"time"
	"mnist_example/MNISTLoader"
)

type Network struct {
	LayersNumber int
	Sizes        []int
	Biases       [][]float64
	Weights      [][][]float64
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
		Biases:       randBiases(sizes),
		Weights:      randWeights(sizes),
	}
}

func randBiases(sizes []int) [][]float64 {
	biases := make([][]float64, len(sizes)-1)
	for n := range biases {
		biases[n] = make([]float64, sizes[n+1])

		for m := range biases[n] {
			s1 := rand.NewSource(time.Now().UnixNano())
			r1 := rand.New(s1)
			biases[n][m] = r1.Float64()*2 - 1
		}
	}
	return biases
}

func randWeights(sizes []int) [][][]float64 {
	weights := make([][][]float64, len(sizes)-1)
	for n := range weights {
		weights[n] = make([][]float64, sizes[n+1])

		for m := range weights[n] {
			weights[n][m] = make([]float64, sizes[n])

			for j := range weights[n][m] {
				s1 := rand.NewSource(time.Now().UnixNano())
				r1 := rand.New(s1)
				w := r1.Float64()*2 - 1
				weights[n][m][j] = w
			}

		}
	}
	return weights
}

func (net *Network) FeedForward(inputs []float64) []float64 {
	prev := inputs
	var values []float64
	for layer, size := range net.Sizes[1:] {
		values = make([]float64, size)
		for n := range values {
			var result float64 = 0
			for m, v := range prev {
				result += net.Weights[layer][n][m] * v
			}
			w := sigmoid(result + net.Biases[layer][n])
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
	nablaBiases := net.Biases
	nablaWeights := net.Weights
	//for l, b := range nablaBiases {
	//	for j := range b {
	//		nablaBiases[l][j] = 0
	//	}
	//}
	//for l1, w1 := range nablaWeights {
	//	for l2, w2 := range w1 {
	//		for l3 := range w2 {
	//			nablaWeights[l1][l2][l3] = 0
	//		}
	//	}
	//}

	for _, m := range miniBatch {
		w, b := net.Backprop(m.Data, m.matrixValue())

		for l1, b1 := range nablaBiases {
			for l2 := range b1 {
				nablaBiases[l1][l2] += b[l1][l2]
			}
		}
		for l1, w1 := range nablaWeights {
			for l2, w2 := range w1 {
				for l3 := range w2 {
					nablaWeights[l1][l2][l3] += w[l1][l2][l3]
				}
			}
		}
	}

	for l1, b1 := range net.Biases {
		for l2, b2 := range b1 {
			net.Biases[l1][l2] = b2 - eta/float64(len(miniBatch))*nablaBiases[l1][l2]
		}
	}
	for l1, w1 := range net.Weights {
		for l2, w2 := range w1 {
			for l3, w3 := range w2 {
				net.Weights[l1][l2][l3] = w3 - eta/float64(len(miniBatch))*nablaWeights[l1][l2][l3]
			}
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

func (net *Network) Backprop(x []float64, y []float64) ([][][]float64, [][]float64) {
	var nablaBiases [][]float64
	for _, l := range net.Biases {
		nablaBiases = append(
			nablaBiases,
			make([]float64, len(l)),
		)
	}
	var nablaWeights [][]float64
	for _, l := range net.Weights {
		nablaWeights = append(
			nablaWeights,
			make([]float64, len(l)),
		)
	}

	activation := x
	activations := [][]float64{
		activation,
	}
	var zs [][]float64

	for l, b := range net.Biases {
		w := net.Weights[l]
		z := make([]float64, len(b))

		for n := range z {
			var v float64

			for j, a := range activation {
				v += a * w[n][j]
			}
			v += b[n]

			z[n] = v
		}

		zs = append(zs, z)
		activation = sigmoidM(z)

		activations = append(activations, activation)
	}

	deltaWeight := net.Weights
	deltaBiases := net.Biases

	for n, a := range activations[len(activations)-1] {
		d := (a - y[n]) * a * (1 - a)
		deltaBiases[len(deltaWeight)-1][n] = d
		for j, a2 := range activations[len(activations)-2] {
			deltaWeight[len(deltaWeight)-1][n][j] = d * a2
		}
	}

	for l := len(deltaWeight) - 2; l >= 0; l-- {
		dw := deltaWeight[l]
		for n, weights := range dw {
			sp := sigmoidPrime(zs[l][n])

			var e float64
			for j, w := range deltaWeight[l+1] {
				e += w[n] * sigmoidPrime(zs[l+1][j])
			}

			deltaBiases[l][n] = sp * e

			for j := range weights {
				deltaWeight[l][n][j] = deltaBiases[l][n] * activations[l][j]
			}
		}
	}

	return deltaWeight, deltaBiases
}

func sigmoidPrime(z float64) float64 {
	return sigmoid(z) * (1 - sigmoid(z))
}

func sigmoid(x float64) (s float64) {
	return 1 / ( 1 + float64(math.Pow(math.E, -x)))
}

func sigmoidM(x []float64) (s []float64) {
	for _, v := range x {
		s = append(s, 1/( 1+float64(math.Pow(math.E, -v))))
	}

	return
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
	images, labels := MNISTLoader.LoadTrain("/Users/uffywen/uffy-go/src/mnist_example/data")

	var data []MNIST

	for n, image := range images {
		label := labels[n]

		inputs := make([]float64, 784)
		for n, p := range image {
			inputs[n] = float64(p)
		}

		data = append(data, MNIST{
			Data:  inputs,
			Value: int(label),
		})
	}

	net := NewNetwork([]int{784, 30, 10})
	loss := net.TotalLoss(data)

	fmt.Println("current net total lose: ", loss)

	net.SGD(data, 10, 10, 3)
}

func matrixToInt(x []float64) int {
	max := 0
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
