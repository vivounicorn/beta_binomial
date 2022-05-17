// Copyright (c) 2018, Jack Parkinson. All rights reserved.
// Use of this source code is governed by the BSD 3-Clause
// license that can be found in the LICENSE file.

package special_test

import (
	"testing"

	. "scientificgo.org/special"
	"scientificgo.org/testutil"
)

var casesChebyshevU = []struct {
	Label    string
	In1      int
	In2, Out float64
}{
	{"", 1, nan, nan},
	{"", 9, +inf, +inf},
	{"", 8, -inf, +inf},
	{"", 9, -inf, -inf},
	{"", 3, 0, 0},
	{"", 10, 0, -1},
	{"", 11, 1, 12},
	{"", 11, -1, -12},
	{"", 0, -10, 1},
	{"", -1, -10, 0},
	{"", 1, -10, -20},
	{"", -3, -10, 20},
	{"", 7, 5.653, 2.2519643853535446868221009536e7},
	{"", 8, 0.5, 0},
	{"", 20, 0.5, 0},
	{"", 245670, 0.5, 1},
	{"", 25, 0.5653, 0.105630867205347272993912993337121475070982482742753766493},
	{"", 30, 0.5653, -1.18177274966401757535034244348459911795334580348773313111},
	{"", 35, 0.5653, -0.42984855978448930094170152092996456093723793883277198979},
	{"", 40, 0.5653, 1.063844401551898424263404702891522637719226917792708881171},
	{"", 45, 0.5653, 0.721712775838257655684186817270448626911644083898522515063},
	{"", 50, 0.5653, -0.86584351605704141337582479184684637573379523593343539188},
	{"", 55, 0.5653, -0.95925572729862559356220671466378093788234830630883282836},
	{"", 60, 0.5653, 0.602673055757878362533262379941036844843258229786532292707},
	{"", 65, 0.5653, 1.124598232930941201752002058361094321608390848492674516388},
	{"", 70, 0.5653, -0.29414111147069492240964626250844884272839686914039346154},
	{"", 45, 5.653, 1.7686985723826779106719444610315516285595747943381758e+47},
	{"", 46, 5.653, 1.9839223748822886791906710760502109724856533950369382e+48},
	{"", 50, 5.653, 3.1405670960834079758171585620307991794780247203737677e+52},
	{"", 55, 5.653, 5.5765079697637498211781120028940424070932320832542064e+57},
	{"", 60, 5.653, 9.9018553609697263342068339928657930451421065028266957e+62},
	{"", 65, 5.653, 1.7582103373864394291770183488355909200781115790457421e+68},
	{"", 70, 5.653, 3.1219437951775878656607929222332593137931640196471191e+73},
	{"", 19, 2e+08, 2.7487790694399996907623546880000146028888063999996241e+163},
	{"", 67, 1.5e+04, 9.2709456349204110975326780846673773440744578594774910e+299},
}

func TestChebyshevU(t *testing.T) { testutil.Test(t, tol, casesChebyshevU, ChebyshevU) }

/*
func BenchmarkChebyshevU(b *testing.B) {
	GlobalF = bench(b, cChebyshevU, "",
		func(x []float64) float64 {
			return ChebyshevU(int(x[0]), x[1])
		})
}
*/