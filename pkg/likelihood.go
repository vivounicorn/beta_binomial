package pkg

//
// Likelihood
// @Description: 计算似然函数接口
//
type Likelihood interface {
	getGradient(x []float64) []float64

	getHessian(x []float64) [][]float64
}
