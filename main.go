package main

import (
	"beta_binomial/pkg"
	"fmt"
	"strconv"
	"strings"
)

func main() {

	method := pkg.NewtonMethod{}
	newton := method.Initialize()
	newton.SetMaxits(200000)
	newton.SetXstart([]float64{100000, 1000000})
	newton.SetDebug(true)
	newton.SetFrequency(1000)

	candidate := "1:0,2:0,3:0,4:0,2:1,3:1,10:2"
	split1 := strings.Split(candidate, ",")
	length := len(split1)

	views := make([]float64, length)
	clicks := make([]float64, length)

	for i := 0; i < length; i++ {
		split2 := strings.Split(split1[i], ":")
		views[i], _ = strconv.ParseFloat(split2[0], 64)
		clicks[i], _ = strconv.ParseFloat(split2[1], 64)
	}

	data := newton.SolveNewtonMethod(views, clicks)
	if data == nil {
		fmt.Println("no solution for B-B distribution parameters estimation.")
	}

}
