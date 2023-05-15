package main

import (
	"math"
	"math/rand"
	"time"

	"github.com/EganBoschCodes/lossless/datasets"
	"github.com/EganBoschCodes/lossless/neuralnetworks/layers"
	"github.com/EganBoschCodes/lossless/neuralnetworks/networks"
)

func GetSpiralDataset() []datasets.DataPoint {
	points := make([]datasets.DataPoint, 0)

	for r := 0.2; r < 3; r += 0.05 {
		p1 := datasets.DataPoint{Input: []float64{r * math.Sin(r), r * math.Cos(r)}, Output: []float64{1, 0, 0}}
		p2 := datasets.DataPoint{Input: []float64{r * math.Sin(r+2.049), r * math.Cos(r+2.049)}, Output: []float64{0, 1, 0}}
		p3 := datasets.DataPoint{Input: []float64{r * math.Sin(r-2.049), r * math.Cos(r-2.049)}, Output: []float64{0, 0, 1}}

		points = append(points, p1, p2, p3)
	}

	rand.Shuffle(len(points), func(i, j int) { points[i], points[j] = points[j], points[i] })

	return points
}

func main() {
	mlp := networks.Perceptron{}

	mlp.Initialize(2,
		&layers.LinearLayer{Outputs: 7},
		&layers.TanhLayer{},
		&layers.LinearLayer{Outputs: 3},
		&layers.SoftmaxLayer{},
	)

	mlp.BatchSize = 32
	mlp.LearningRate = 1

	data := GetSpiralDataset()
	trainingData, testingData := data[:120], data[120:]

	mlp.Train(trainingData, testingData, 10*time.Second)

	mlp.Save("savednetworks", "MyMLP")
	mlp.PrettyPrint("savednetworks", "MyMLP")
}
