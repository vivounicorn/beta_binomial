package pkg

import (
	"fmt"
	_ "fmt"
	"math"
)

const (
	TOLERANCE           = 1e-5 // 梯度函数容忍值.
	OBJECTIONTOLERANCE_ = 1e-8 // 损失函数容忍值.
	MU_HESSIAN          = 1e-6 //  Hessian矩阵容忍值.
)

//
// NewtonMethod
// @Description: 采用牛顿法估计Beta-Binonmial分布的alpha及beta参数.
//
type NewtonMethod struct {
	// beta binomial分布的似然函数.
	likelihood *BetaBinomialLikelihood
	maxits     int       //优化最大迭代次数. 2000
	xstart     []float64 //alpha 和 beta的初始值. { 10, 10 }
	frequency  int       //设置调试输出频率. 1000
	debug      bool      //是否打印调试信息. false
	sumPasses  float64   // 信审总次数 0.0
	sumBads    float64   // 逾期总次数 0.0
}

//
// Initialize
// @Description: 初始化函数.
// @receiver bb
// @return *NewtonMethod
//
func (bb *NewtonMethod) Initialize() *NewtonMethod {
	return &NewtonMethod{
		maxits:    2000,
		xstart:    []float64{10, 10},
		frequency: 1000,
		debug:     false,
		sumPasses: 0.0,
		sumBads:   0.0,
	}
}

//
// SetMaxits
// @Description: 设置最大迭代次数.
// @receiver bb
// @param maxits
//
func (bb *NewtonMethod) SetMaxits(maxits int) {
	bb.maxits = maxits
}

//
// SetXstart
// @Description: 设置初始点.
// @receiver bb
// @param xstart
//
func (bb *NewtonMethod) SetXstart(xstart []float64) {
	if len(xstart) != 2 {
		fmt.Println("start point of alpha and beta is error")
	}

	bb.xstart = xstart
}

//
// SetFrequency
// @Description: 设置打印频率.
// @receiver bb
// @param frequency
//
func (bb *NewtonMethod) SetFrequency(frequency int) {
	bb.frequency = frequency
}

//
// SetDebug
// @Description: 设置是否debug.
// @receiver bb
// @param debug
//
func (bb *NewtonMethod) SetDebug(debug bool) {
	bb.debug = debug
}

//
// doAffineSolve
// @Description: 求解仿射函数 A*x=b问题
// @receiver bb
// @param A
// @param rhs
// @return []float64
//
func (bb *NewtonMethod) doAffineSolve(A [][]float64, rhs []float64) []float64 {

	a := A[0][0]
	b := A[0][1]
	c := A[1][0]
	d := A[1][1]
	x := rhs[0]
	y := rhs[1]

	disc := a*d - b*c
	return []float64{(d*x - b*y) / disc, (a*y - c*x) / disc}
}

//
// getEigenvalues
// @Description: 求解2x2 Hessian 矩阵的特征值
// @receiver bb
// @param A
// @return []float64
//
func (bb *NewtonMethod) getEigenvalues(A [][]float64) []float64 {

	a := A[0][0]
	b := A[0][1]
	c := A[1][0]
	d := A[1][1]

	temp := math.Sqrt(a*a + 4*b*c - 2*a*d + d*d)
	e1 := .5 * (a + d - temp)
	e2 := .5 * (a + d + temp)
	return []float64{e1, e2}
}

//
// SolveNewtonMethod
// @Description: 使用牛顿法估计Beta-Binomial分布的参数 alpha 和 beta.
// @receiver bb
// @param passes
// @param bads
// @return []float64
//
func (bb *NewtonMethod) SolveNewtonMethod(passes []float64, bads []float64) []float64 {

	if passes == nil || bads == nil || len(passes) == 0 || len(bads) == 0 || len(passes) != len(bads) {

		fmt.Println("passes and bads data error")
		return nil
	}

	method := BetaBinomialLikelihood{}
	bb.likelihood = method.Initialize(bads, passes)

	var eigenvalues []float64
	var H [][]float64
	p := []float64{0, 0}
	x := []float64{0, 0}
	x0 := []float64{bb.xstart[0], bb.xstart[1]}
	G := bb.likelihood.getGradient(x0)
	correction := 0.0
	a := 0.0
	i := 0
	for i = 0; i < bb.maxits; i++ {
		// 使用单位矩阵确保牛顿方向是下降方向
		H = bb.likelihood.getHessian(x0)
		eigenvalues = bb.getEigenvalues(H)
		correction = math.Min(eigenvalues[0], eigenvalues[1])
		correction = math.Min(0, correction)
		H[0][0] += MU_HESSIAN - correction
		H[1][1] += MU_HESSIAN - correction

		// H*(x0-x) = G
		p = bb.doAffineSolve(H, G)
		a = 1
		x[0] = x0[0] - p[0]
		x[1] = x0[1] - p[1]

		// 开始line search 寻找优化方向.
		for true {
			if !(a > 1e-16 && (bb.likelihood.getObjectFunction(x)-bb.likelihood.getObjectFunction(x0) > 0 || x[0] <= 0 || x[1] <= 0)) {
				break
			}

			a /= 2
			x[0] = x0[0] - a*p[0]
			x[1] = x0[1] - a*p[1]
		}

		G = bb.likelihood.getGradient(x)
		if (math.Abs(G[0]) < TOLERANCE && math.Abs(G[1]) < TOLERANCE) || math.Max(math.Abs(x[0]-x0[0]), math.Abs(x[1]-x0[1])) < OBJECTIONTOLERANCE_ {
			break
		}

		x0 = []float64{x[0], x[1]}

		if bb.debug && i%bb.frequency == 0 {
			fmt.Printf(
				"current iteration:%d:(alpha=%f,beta=%f).\r\n\n", i,
				x[0], x[1])
		}
	}

	if i == bb.maxits {
		fmt.Println("not converge with maximum number of iterations")
	}

	for j := 0; j < len(passes); j++ {
		bb.sumPasses += passes[j]
		bb.sumBads += bads[j]
	}

	// 打印调试信息.
	if bb.debug {
		fmt.Printf("\nStatistic information details: average ctr=%f. (total page views=%d, total clicks=%d)\r\n",
			bb.sumBads/bb.sumPasses, bb.sumPasses, bb.sumBads)
		fmt.Printf("Beta-Binomial distribution: smooth ctr=%f. (alpha=%f, beta=%f)\r\n",
			x[0]/(x[0]+x[1]), x[0], x[1])
	}

	return x
}
