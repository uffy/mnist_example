package classifier

import (
	"fmt"
	"math"
	"math/rand"
	"time"
	"mnist_example/MNISTLoader"
	"sync"
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

func zeroBiases(sizes []int) [][]float64 {
	biases := make([][]float64, len(sizes)-1)
	for n := range biases {
		biases[n] = make([]float64, sizes[n+1])

		for m := range biases[n] {
			biases[n][m] = 0
		}
	}
	return biases
}

func zeroWeights(sizes []int) [][][]float64 {
	weights := make([][][]float64, len(sizes)-1)
	for n := range weights {
		weights[n] = make([][]float64, sizes[n+1])

		for m := range weights[n] {
			weights[n][m] = make([]float64, sizes[n])

			for j := range weights[n][m] {
				weights[n][m][j] = 0
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

func (net *Network) SGD(trainingData []MNIST, testData []MNIST, epochs int, miniBatchSize int, eta float64) {
	trainingData = shuffleMNIST(trainingData)
	n := len(trainingData)
	fmt.Printf("traningData: %d, testData: %d\n", len(trainingData), len(testData))

	for j := 0; j < epochs; j++ {
		for i := 0; i < n; i += miniBatchSize {
			end := i + miniBatchSize
			if end > n {
				end = n
			}
			net.updateMiniBatch(trainingData[i:end], eta)
		}

		eta *= 0.9
		s, t := net.evaluate(testData)
		fmt.Printf("Epochs %d %d/%d, Total loss: %.2f\n", j, s, t, net.TotalLoss(testData))
	}
}

func (net *Network) OnlineSGD(trainingData []MNIST, testData []MNIST, epochs int, miniBatchSize int, eta float64) {
	trainingData = shuffleMNIST(trainingData)

	for j := 0; j < epochs; j++ {
		for n, d := range trainingData {
			w, b := net.Backprop(d.Data, d.matrixValue())
			for l1 := range net.Weights {
				for l2 := range net.Weights[l1] {
					net.Biases[l1][l2] -= b[l1][l2] * eta
					for l3 := range net.Weights[l1][l2] {
						net.Weights[l1][l2][l3] -= w[l1][l2][l3] * eta
					}
				}
			}

			if n%100 == 0 {
				s, t := net.evaluate(testData)
				fmt.Printf("Epochs %d %d/%d, Total loss: %.2f\n", j, s, t, net.TotalLoss(testData))
			}

			if n%100 == 0 {
				eta *= 0.99
			}
		}

	}
}

func (net *Network) evaluate(data []MNIST) (success int, total int) {
	for _, mnist := range data {
		total++
		a := matrixToInt(net.FeedForward(mnist.Data))

		if a == mnist.Value {
			success++
		}
	}

	return success, total
}

func (net *Network) updateMiniBatch(miniBatch []MNIST, eta float64) {
	nablaBiases := zeroBiases(net.Sizes)
	nablaWeights := zeroWeights(net.Sizes)

	wg := sync.WaitGroup{}
	lock := sync.Mutex{}
	for _, m := range miniBatch {
		wg.Add(1)

		go func() {
			defer wg.Done()
			w, b := net.Backprop(m.Data, m.matrixValue())

			lock.Lock()
			defer lock.Unlock()
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
		}()
	}
	wg.Wait()

	pEta := eta / float64(len(miniBatch))
	for l1, b1 := range net.Biases {
		for l2 := range b1 {
			net.Biases[l1][l2] -= pEta * nablaBiases[l1][l2]
		}
	}
	for l1, w1 := range net.Weights {
		for l2, w2 := range w1 {
			for l3 := range w2 {
				net.Weights[l1][l2][l3] -= pEta * nablaWeights[l1][l2][l3]
			}
		}
	}
}

func (net *Network) TotalLoss(batches []MNIST) float64 {
	var total float64 = 0
	for _, batch := range batches {
		value := net.FeedForward(batch.Data)
		L := CSC(batch.matrixValue(), value)
		total += L
	}

	return total * -1 / float64(len(batches))
}

func (net *Network) Backprop(x []float64, y []float64) ([][][]float64, [][]float64) {
	activation := x
	activations := [][]float64{
		activation,
	}

	for l, size := range net.Sizes[1:] {
		w := net.Weights[l]
		z := make([]float64, size)
		for n := range z {
			var v float64

			for j, a := range activation {
				v += a * w[n][j]
			}
			v += net.Biases[l][n]

			z[n] = v
		}

		activation = sigmoidM(z)

		activations = append(activations, activation)
	}

	deltaWeight := zeroWeights(net.Sizes)
	deltaBiases := zeroBiases(net.Sizes)
	le := len(net.Sizes) - 1

	// layer 2, output layer
	for n, a := range activations[le] {
		//CEC
		d := a - y[n]
		//d := (a - y[n]) * a

		// MSE
		//d := (a - y[n]) * a * (1 - a)
		deltaBiases[le-1][n] = d
	}

	// layer 1, hidden layer
	for j, s := range net.Sizes[1:le] {
		l := j + 1
		for n := 0; n < s; n++ {
			sp := sigmoidPrime(activations[l][n]) // d out / d net,

			var d float64
			for m, b := range deltaBiases[l] {
				d += b * net.Weights[l][m][n]
			}

			deltaBiases[l-1][n] = d * sp
		}
	}

	for l1 := range deltaWeight {
		for l2 := range deltaWeight[l1] {
			for l3 := range deltaWeight[l1][l2] {
				deltaWeight[l1][l2][l3] = deltaBiases[l1][l2] * activations[l1][l3]
			}
		}
	}

	return deltaWeight, deltaBiases
}

func sigmoidPrime(z float64) float64 {
	return z * (1 - z)
}

func sigmoid(x float64) (s float64) {
	return 1 / ( 1 + math.Exp(-x))
}

func sigmoidM(x []float64) (s []float64) {
	for _, v := range x {
		s = append(s, sigmoid(v))
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

// cross-entropy cost function
func CSC(a []float64, b []float64) (cost float64) {
	for n := range a {
		var v1, v2 float64
		if b[n] != 0 {
			v1 = math.Log(b[n]) * a[n]
		}
		if b[n] != 1 {
			v2 = math.Log(1-b[n]) * (1 - a[n])
		}

		cost += v1 + v2
	}
	return
}

func (m *MNIST) matrixValue() []float64 {
	v := make([]float64, 10)
	v[m.Value] = 1

	return v
}

var net = NewNetwork([]int{784, 30, 10})

func StartTrain() {
	data := toMNIST(MNISTLoader.LoadTrain("/Users/uffywen/uffy-go/src/mnist_example/data"))
	testData := toMNIST(MNISTLoader.LoadTest("/Users/uffywen/uffy-go/src/mnist_example/data"))

	net.SGD(data, testData, 30, 10, 0.5)
}

func FeedForward(inputs []float64) int {
	v := net.FeedForward(inputs)
	return matrixToInt(v)
}

func toMNIST(images [][]float64, labels []float64) []MNIST {
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

	return data
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
