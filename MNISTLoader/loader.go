package MNISTLoader

import (
	"encoding/binary"
	"errors"
	"log"
	"os"
)

// ReadLabels parses input label file from given path and returns list of MNIST labels and number of distinct labels
func ReadLabels(path string) ([]float64, int, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, 0, err
	}

	var magic int32
	var itemCount int32
	var item byte
	if err := binary.Read(f, binary.BigEndian, &magic); err != nil || magic != 2049 {
		return nil, 0, errors.New("mnistloader: cannot read magic number in label file")
	}
	if err := binary.Read(f, binary.BigEndian, &itemCount); err != nil {
		return nil, 0, errors.New("mnistloader: cannot read count of labels")
	}
	items := make([]float64, itemCount)
	distinct := make(map[float64]bool)
	for i := int32(0); i < itemCount; i++ {
		if err := binary.Read(f, binary.BigEndian, &item); err != nil {
			return nil, 0, errors.New("mnistloader: cannot read label")
		}
		items[i] = float64(item)
		distinct[float64(item)] = true
	}
	return items, len(distinct), nil
}

// ReadImages parses input image file and returns list of normalized MNIST images and length of one input value
func ReadImages(path string) ([][]float64, int, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, 0, err
	}

	var magic int32
	var itemCount int32
	var rows int32
	var cols int32
	if err := binary.Read(f, binary.BigEndian, &magic); err != nil || magic != 2051 {
		return nil, 0, errors.New("mnistloader: cannot read magic number in images file")
	}
	if err := binary.Read(f, binary.BigEndian, &itemCount); err != nil {
		return nil, 0, errors.New("mnistloader: cannot read count of images")
	}
	if err := binary.Read(f, binary.BigEndian, &rows); err != nil {
		return nil, 0, errors.New("mnistloader: cannot read number of rows of images")
	}
	if err := binary.Read(f, binary.BigEndian, &cols); err != nil {
		return nil, 0, errors.New("mnistloader: cannot read number of cols of images")
	}
	items := make([][]float64, itemCount)
	for i := int32(0); i < itemCount; i++ {
		item := make([]byte, rows*cols)
		items[i] = make([]float64, rows*cols)
		if err := binary.Read(f, binary.BigEndian, item); err != nil {
			return nil, 0, errors.New("mnistloader: cannot read image")
		}
		for j, val := range item {
			items[i][j] = float64(val) / 256.0 // normalize
		}
	}
	return items, int(rows * cols), nil
}

func LoadTrain(path string) ([][]float64, []float64) {
	images, _, err := ReadImages(path + "/train-images-idx3-ubyte")
	if err != nil {
		log.Fatal(err)
	}
	labels, _, err := ReadLabels(path + "/train-labels-idx1-ubyte")

	if err != nil {
		log.Fatal(err)
	}

	return images, labels
}
