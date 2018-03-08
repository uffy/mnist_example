package main
//
//import (
//	"fmt"
//	"github.com/petar/goMNIST"
//	"log"
//	"math"
//)
//
//var total = 0
//var successTotal = 0
//
//func main() {
//	train, test, err := GoMNIST.Load("./data")
//
//	if err != nil {
//		log.Fatal(err)
//	}
//
//	sweeper := train.Sweep()
//
//	layouts = make([]*Layout, 3)
//	net_numbers := []int{784, 15, 10}
//
//	for i, n := range net_numbers {
//		layouts[i] = &Layout{
//			nets: make([]*Net, n),
//		}
//
//		if i == len(net_numbers)-1 {
//			continue
//		}
//
//		for m := range layouts[i].nets {
//			layouts[i].nets[m] = &Net{
//				weights: make([]float32, net_numbers[i+1]),
//			}
//		}
//	}
//
//	i := 0
//	for {
//		image, label, present := sweeper.Next()
//		if !present {
//			break
//		}
//
//		inputs := make([]float32, 784)
//
//		for n, p := range image {
//			inputs[n] = float32(p)
//		}
//
//		total++
//		trainWeights(layouts,
//			inputs,
//			matrix(int(label), 10),
//			int(label),
//		)
//
//		i++
//		if i > 1000 {
//			break
//		}
//	}
//
//	sweeper = test.Sweep()
//	for {
//		image, label, present := sweeper.Next()
//		if !present {
//			break
//		}
//
//		var inputs []float32 = make([]float32, 784)
//
//		for n, p := range image {
//			inputs[n] = float32(p)
//		}
//
//		fmt.Println(matrixToInt(calc(inputs)), label)
//	}
//
//}
//
//type Net struct {
//	weights []float32
//}
//type Layout struct {
//	nets []*Net
//}
//
//type Layouts []*Layout
//
//var layouts Layouts
//
//func calc(inputs []float32) []float32 {
//	var prev []float32
//	var values []float32
//	for layout := range layouts {
//		if layout == 0 {
//			prev = inputs
//			continue
//		}
//
//		values = make([]float32, len(layouts[layout].nets))
//		for n := range values {
//			var result float32 = 0
//			for m := range prev {
//				result += layouts[layout-1].nets[m].weights[n] * prev[m]
//			}
//			values[n] = sigmoid(result)
//		}
//		prev = values
//	}
//
//	return prev
//}
//
//func trainWeights(layouts Layouts, inputs []float32, result []float32, label int) {
//
//	rs := calc(inputs)
//
//	if matrixToInt(rs) == label {
//		successTotal++
//		fmt.Printf("success: %.2f%% \n", float32(successTotal)/float32(total)*100)
//	}
//
//	for l := range layouts {
//
//		if l == len(layouts)-1 {
//			continue
//		}
//
//		for n := range layouts[l].nets {
//			for w := range layouts[l].nets[n].weights {
//				prevMSE := MSE(calc(inputs), result)
//				prevWeight := layouts[l].nets[n].weights[w]
//				p := 2
//
//				for {
//					layouts[l].nets[n].weights[w] += float32(p-3) * 0.1
//
//					mse := MSE(calc(inputs), result)
//					//fmt.Println(l, n, w, p, prevMSE, mse)
//
//					if prevMSE > mse {
//						prevMSE = mse
//						prevWeight = layouts[l].nets[n].weights[w]
//					} else {
//						p--
//						layouts[l].nets[n].weights[w] = prevWeight
//					}
//
//					if p == 0 {
//						break
//					}
//				}
//
//			}
//		}
//	}
//}
//
//func sigmoid(x float32) float32 {
//	return 1 / ( 1 + float32(math.Pow(math.E, -float64(x))));
//}
//
//func matrix(v int, len int) []float32 {
//	m := make([]float32, len)
//	m[v] = 1
//
//	return m
//}
//func matrixToInt(x []float32) int {
//	max := 0
//
//	for n, v := range x {
//		if v > x[max] {
//			max = n
//		}
//	}
//
//	return max
//}
//
//func MSE(a []float32, b []float32) (cost float32) {
//	for n := range a {
//		cost += float32(math.Pow(float64(a[n]-b[n]), 2)) / 2
//	}
//	return
//}
//
//func Oss(x []float32, n int, rs []float32) {
//
//}
