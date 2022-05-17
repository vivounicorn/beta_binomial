// Copyright (c) 2018, Jack Parkinson. All rights reserved.
// Use of this source code is governed by the BSD 3-Clause
// license that can be found in the LICENSE file.

package special_test

import (
	"testing"

	. "scientificgo.org/special"
	"scientificgo.org/testutil"
)

var casesPolygamma = []struct {
	Label    string
	In1      int
	In2, Out float64
}{
	{"", 0, 1e+08, 18.420680738952367},
	{"", 1, 1e+08, 1.000000005e-08},
	{"", 2, nan, nan},
	{"", 2, -inf, nan},
	{"", 2, +inf, 0},
	{"", 2, -2, nan},
	{"", 2, 1e+08, -1.00000001e-16},
	{"", 2, 1e-08, -2e+24},
	{"", 2, 10, -0.011049834970802067},
	{"", 2, 6, -0.0327897322451145},
	{"", 2, 7, -0.023530472985855238},
	{"", 2, 8, -0.017699569195767775},
	{"", 2, 4.9, -0.05100493156907858},
	{"", 2, -4.9, -1998.6936655466138},
	{"", 2, -10.2, 247.03920499486188},
	{"", 3, 1e+08, 2.0000000300000002e-24},
	{"", 3, 1e-08, 6e+32},
	{"", 3, -1e-08, 6e+32},
	{"", 3, 10, 0.0023199013042898686},
	{"", 3, -10.2, 3768.6142614155146},
	{"", 3, 4.9, 0.022897748562742785},
	{"", 3, -4.9, 60014.239401127015},
	{"", 3, -20.2, 3768.6156617121133},
	{"", 3, -110.2, 3768.6158854609644},
	{"", 4, 1e-08, -2.4e+41},
	{"", 4, -1e-08, 2.4e+41},
	{"", 4, 0.5, -771.4742498266672},
	{"", 4, -0.5, -3.4742498266672253},
	{"", 4, -13.9, -2.3999738348960066e+06},
	{"", 4, -14.01, 2.3999999999755655e+11},
	{"", 4, 14.1, -0.00017460126158302312},
	{"", 4, 15.9, -0.00010630348495697623},
	{"", 4, 16, -0.00010359125360358783},
	{"", 4, -16.01, 2.3999999999736902e+11},
	{"", 4, 17, -8.070307000983783e-05},
	{"", 5, 1e-08, 1.2e+50},
	{"", 5, -1e-08, 1.2e+50},
	{"", 5, 0.5, 7691.113548602436},
	{"", 5, -0.5, 15371.113548602436},
	{"", 5, 5, 0.01226150963595438},
	{"", 5, 6, 0.004581509635954379},
	{"", 5, 7, 0.0020094931750490293},
	{"", 5, 8, 0.0009895100047713388},
	{"", 5, -5.1, 1.2000029790471852e+08},
	{"", 5, -13.9, 1.2000029790887256e+08},
	{"", 5, 14.1, 5.123862252375418e-05},
	{"", 5, 15.9, 2.7563111256905805e-05},
	{"", 5, 16, 2.6687171525751197e-05},
	{"", 5, 17, 1.9534614152704322e-05},
	{"", 6, 0.001, -7.2e+23},
	{"", 6, 1, -726.0114797149845},
	{"", 6, 2, -6.011479714984436},
	{"", 6, 5, -0.013316295488550551},
	{"", 6, 10, -0.0001601508710767886},
	{"", 9, -10.5, 7.431909047701783e+08},
	{"", 9, -100.1, 3.628800001181459e+15},
	{"", 10, 10, -5.7675966863222595e-05},
	{"", 20, 10, -0.0028271335649279823},
	{"", 20, 11, -0.0003942315567513425},
	{"", 20, -20.2, 1.1600980797653613e+33},
	{"", 30, 15, -0.00010699799160744993},
	{"", 50, 10, -3.065245725290585e+13},
	{"", 50, -10.5, -2.475965723385829e+10},
	{"", 100, 0.1, -9.332621544394415e+258},
	{"", 100, 10, -9.333237300479084e+56},
	{"", 100, -10.5, -6.9122173544603244e+50},
	{"", 100, -100.5, -3.3121910883047074e-45},
	{"", 150, 15, -1.4693955875276522e+85},
	{"", 200, 50, -2.5829713930705358e+33},
	{"", 150, -0.001, +inf},
	{"", 6, 25, -5.532496491858543e-07},
	{"", 6, 50, -8.151546844438303e-09},
	{"", 6, 1000, -1.20360419999496e-16},
	{"", 7, 1000, 7.2252335999496e-19},
	{"", 8, 1000, -5.06019023994456e-21},
	{"", 9, 1000, 4.050174239933472e-23},
	{"", 10, 1000, -3.646977263913514e-25},
	{"", 11, 1000, 3.6487983166789194e-27},
	{"", 11, 100, 3.8323744698817625e-16},
	{"", 111, 1, 1.7629525510902446e+180},
}

func TestPolygamma(t *testing.T) { testutil.Test(t, tol, casesPolygamma, Polygamma) }

/*
func BenchmarkPolygamma(b *testing.B) {
	GlobalF = bench(b, cPolygamma, "",
		func(x []float64) float64 {
			return Polygamma(int(x[0]), x[1])
		})
}
*/
