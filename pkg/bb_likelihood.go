package pkg

import (
	"beta_binomial/third_party/special"
	_ "beta_binomial/third_party/special"
	"math"
	_ "math"
)

//
// LAMBDA
// @Description: 正则项惩罚系数.
//
const (
	LAMBDA = 1e-6
)

//
// BetaBinomialLikelihood
// @Description: 计算 beta binomial 分布的似然函数.
//
type BetaBinomialLikelihood struct {
	//
	// bads
	// @Description: 逾期的订单.
	//
	bads []float64

	//
	// passes
	// @Description: 信审通过的订单.
	//
	passes []float64
}

//
// Initialize
// @Description: 初始化函数.
// @param bads
// @param passes
// @return *BetaBinomialLikelihood
//
func (bb *BetaBinomialLikelihood) Initialize(bads []float64, passes []float64) *BetaBinomialLikelihood {

	return &BetaBinomialLikelihood{
		bads:   bads,
		passes: passes,
	}
}

//
// callLogGamma
// @Description: 调用 log gamma函数.
// @param x
// @return float64
//
func (bb *BetaBinomialLikelihood) callLogGamma(x float64) float64 {
	res, _ := math.Lgamma(x)
	return res
}

//
// callDigamma
// @Description: 调用 digamma 函数.
// @param x
// @return float64
//
func (bb *BetaBinomialLikelihood) callDigamma(x float64) float64 {
	return special.Digamma(x)
}

//
// callTrigamma
// @Description: 调用trigamma 函数.
// @param x
// @return float64
//
func (bb *BetaBinomialLikelihood) callTrigamma(x float64) float64 {
	return special.Trigamma(x)
}

//
// getLogLikelihood
// @Description: 计算Beta Binomial分布的log likelihood函数.
// @receiver bb
// @param alpha
// @param beta
// @return float64
//
func (bb *BetaBinomialLikelihood) getLogLikelihood(alpha float64, beta float64) float64 {
	if alpha <= 0 || beta <= 0 {
		return math.Inf(-1)
	}

	total := 0.0
	for i := 0; i < len(bb.passes); i++ {
		total += bb.callLogGamma(alpha+beta) - bb.callLogGamma(alpha) - bb.callLogGamma(beta) + bb.callLogGamma(alpha+bb.bads[i]) + bb.callLogGamma(beta+bb.passes[i]-bb.bads[i]) - bb.callLogGamma(alpha+beta+bb.passes[i])
	}
	return total
}

//
// getLogLikelihoodGradient
// @Description: 计算Beta Binomial分布的log likelihood函数的梯度值.
// @receiver bb
// @param alpha
// @param beta
// @return []float64
//
func (bb *BetaBinomialLikelihood) getLogLikelihoodGradient(alpha float64, beta float64) []float64 {
	Da := 0.0
	Db := 0.0
	for i := 0; i < len(bb.passes); i++ {
		Da += bb.callDigamma(alpha+beta) - bb.callDigamma(alpha) + bb.callDigamma(alpha+bb.bads[i]) - bb.callDigamma(alpha+beta+bb.passes[i])

		Db += bb.callDigamma(alpha+beta) - bb.callDigamma(beta) + bb.callDigamma(beta+bb.passes[i]-bb.bads[i]) - bb.callDigamma(alpha+beta+bb.passes[i])
	}
	return []float64{Da, Db}
}

//
// getLogLikelihoodHessian
// @Description: 计算Beta Binomial分布的log likelihood函数的Hessian矩阵.
// @receiver bb
// @param alpha
// @param beta
// @return [][]float64
//
func (bb *BetaBinomialLikelihood) getLogLikelihoodHessian(alpha float64, beta float64) [][]float64 {

	Daa := 0.0
	Dab := 0.0
	Dbb := 0.0
	for i := 0; i < len(bb.passes); i++ {
		Daa += bb.callTrigamma(alpha+beta) - bb.callTrigamma(alpha) + bb.callTrigamma(alpha+bb.bads[i]) - bb.callTrigamma(alpha+beta+bb.passes[i])

		Dbb += bb.callTrigamma(alpha+beta) - bb.callTrigamma(beta) + bb.callTrigamma(beta+bb.passes[i]-bb.bads[i]) - bb.callTrigamma(alpha+beta+bb.passes[i])

		Dab += bb.callTrigamma(alpha+beta) - bb.callTrigamma(alpha+beta+bb.passes[i])
	}
	return [][]float64{{Daa, Dab}, {Dab, Dbb}}
}

//
// getRegularization
// @Description: 计算正则项.
// @receiver bb
// @param x
// @return float64
//
func (bb *BetaBinomialLikelihood) getRegularization(x []float64) float64 {
	return 0.5 * LAMBDA * (math.Pow(x[0], 2) + math.Pow(x[1], 2))
}

//
// getRegularizationGradient
// @Description: 计算正则项梯度.
// @receiver bb
// @param x
// @return []float64
//
func (bb *BetaBinomialLikelihood) getRegularizationGradient(x []float64) []float64 {
	return []float64{LAMBDA * x[0], LAMBDA * x[1]}
}

//
// getRegularizationHessian
// @Description: 计算正则项的Hessian矩阵.
// @receiver bb
// @param x
// @return [][]float64
//
func (bb *BetaBinomialLikelihood) getRegularizationHessian(x []float64) [][]float64 {
	return [][]float64{{LAMBDA, 0},
		{0, LAMBDA}}
}

//
// getObjectFunction
// @Description: 计算优化损失函数值
// @receiver bb
// @param x
// @return float64
//
func (bb *BetaBinomialLikelihood) getObjectFunction(x []float64) float64 {
	return -bb.getLogLikelihood(x[0], x[1]) + bb.getRegularization(x)
}

//
// getGradient
// @Description: 计算损失函数梯度向量.
// @receiver bb
// @param x
// @return []float64
//
func (bb *BetaBinomialLikelihood) getGradient(x []float64) []float64 {
	G := bb.getLogLikelihoodGradient(x[0], x[1])
	r := bb.getRegularizationGradient(x)
	return []float64{-G[0] + r[0], -G[1] + r[1]}
}

//
// getHessian
// @Description: 计算损失函数的Hessian矩阵.
// @receiver bb
// @param x
// @return [][]float64
//
func (bb *BetaBinomialLikelihood) getHessian(x []float64) [][]float64 {
	H := bb.getLogLikelihoodHessian(x[0], x[1])
	r := bb.getRegularizationHessian(x)
	return [][]float64{{-H[0][0] + r[0][0], -H[0][1] + r[0][1]},
		{-H[1][0] + r[1][0], -H[1][1] + r[1][1]}}
}
