const assert = require('node:assert/strict');
const {curves, histogram, auc, bootstrapAuc, confusion, rates, calibration, mcnemar} = require('../dashboard/charts.js');
const perfect = [{label: 0, score: .1}, {label: 0, score: .2}, {label: 1, score: .8}, {label: 1, score: .9}];
assert.equal(curves(perfect).ap, 1);
assert.deepEqual(curves(perfect).roc.at(-1), [1, 1]);
const tied = curves([{label: 0, score: .5}, {label: 1, score: .5}]);
assert.deepEqual(tied.roc, [[0, 0], [1, 1]]);
assert.equal(tied.ap, .5);
assert.equal(curves([]), null);
assert.equal(curves([{label: 0, score: .1}]), null);
assert.equal(histogram([{label: 1, score: 1}])[9].fake, 1);
assert.equal(histogram([{label: 0, score: 0}])[0].real, 1);
assert.equal(histogram([{label: 0, score: NaN}]).reduce((n, b) => n + b.real, 0), 0);
assert.equal(curves([...perfect, {label: 1, score: NaN}]).ap, 1);
// ROC-AUC: perfect, inverted, ties count half, single class has none.
assert.equal(auc(perfect), 1);
assert.equal(auc(perfect.map(p => ({...p, score: 1 - p.score}))), 0);
assert.equal(auc([{label: 0, score: .5}, {label: 1, score: .5}]), .5);
assert.equal(auc([{label: 1, score: .3}]), null);
const ci = bootstrapAuc([...perfect, {label: 0, score: .85}, {label: 1, score: .15}], 200, 1);
assert.ok(ci[0] <= ci[1] && ci[0] >= 0 && ci[1] <= 1);
assert.deepEqual(bootstrapAuc(perfect, 50, 3), bootstrapAuc(perfect, 50, 3));
// Rates from a confusion matrix, including MCC.
const r = rates(confusion(perfect, .5));
assert.equal(r.accuracy, 1); assert.equal(r.mcc, 1); assert.equal(r.fpr, 0);
assert.equal(rates({tp: 0, fp: 0, tn: 5, fn: 0}).precision, null);
// Calibration: Brier score and ECE of a perfectly confident, perfectly right model are 0.
const cal = calibration([{label: 0, score: 0}, {label: 1, score: 1}]);
assert.equal(cal.brier, 0); assert.equal(cal.ece, 0);
// McNemar exact test: no discordant pairs gives 1; 10 vs 0 is significant; symmetric.
assert.equal(mcnemar(0, 0), 1);
assert.ok(mcnemar(10, 0) < .01);
assert.equal(mcnemar(3, 7), mcnemar(7, 3));
console.log('Chart calculations passed: ROC/PR, histogram, AUC with bootstrap CI, rates and MCC, calibration, McNemar.');
