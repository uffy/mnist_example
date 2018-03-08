package main

import "mnist_example/MNISTLoader"

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
