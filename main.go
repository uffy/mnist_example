package main

import (
	"mnist_example/classifier"
	"net/http"
	"io/ioutil"
	"log"
	"strings"
	"strconv"
	"runtime"
	"os/exec"
)

func main() {
	go classifier.StartTrain()
	indexHandler := http.FileServer(http.Dir("./public"))

	//open("http://localhost:12306")
	http.ListenAndServe(":12306", http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/":
			indexHandler.ServeHTTP(w, r)
		case "/classifier":
			if body, err := ioutil.ReadAll(r.Body); err != nil {
				log.Fatal(err)
			} else {
				d := strings.Split(string(body), ",")
				var inputs []float64
				for _, s := range d {
					f, _ := strconv.ParseFloat(s, 32)
					inputs = append(inputs, f)
				}
				rs := strconv.Itoa(classifier.FeedForward(inputs))
				w.Write([]byte(rs))
				r.Body.Close()
			}
		}
	}))
}

// open opens the specified URL in the default browser of the user.
func open(url string) error {
	var cmd string
	var args []string

	switch runtime.GOOS {
	case "windows":
		cmd = "cmd"
		args = []string{"/c", "start"}
	case "darwin":
		cmd = "open"
	default: // "linux", "freebsd", "openbsd", "netbsd"
		cmd = "xdg-open"
	}
	args = append(args, url)
	return exec.Command(cmd, args...).Start()
}
