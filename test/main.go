package main

import "fmt"

func quicksort(arr []int) []int {
	if len(arr) < 1 {
		return arr
	}

	pivot := arr[len(arr)/2]

	var left, right, middle []int
	for _, v := range arr {
		if v < pivot {
			left = append(left, v)
		} else if v == pivot {
			middle = append(middle, v)
		} else if v > pivot {
			right = append(right, v)
		}
	}

	return append(quicksort(left), append(middle, quicksort(right)...)...)

}

func main() {
	arr := quicksort([]int{
		3, 6, 8, 10, 1, 2, 1,
	})

	fmt.Println(arr)
}
