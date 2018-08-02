const snn = require('./main.js');

const simple_train = () => {
	// nothing special about this configuration, decided it just like that
	let nn = new snn.Network([4, 10, 5, 1]);

	const SOME_INP1 = [1, 0, 0, 3];
	const SOME_INP2 = [-4, 2, 3.2, -12];

	const loss = (net) => {
		return Math.abs(net.run(SOME_INP1) - net.run(SOME_INP2) * 3);
	}

	// train the network
	for(let i = 0;i < 500;i ++) {
		let copy = snn.copyNN(nn);
		snn.mutate(copy, 0.05);
		if(loss(copy) < loss(nn)) {
			nn = snn.copyNN(copy);
		}
	}

	return loss(nn) < 0.01;
	//nn.run([-12, 4, 34, 12]); // can be absouletly anything
}

const xor_train = (MUTATE_CHANCE) => {
	let nn = new snn.Network([2, 2, 1]);

	const data = [{input: [1, 1], output: [0]}, {input: [1, 0], output: [1]},
	              {input: [0, 1], output: [1]}, {input: [0, 0], output: [0]}];

	const loss = (net, p) => {
		let total = 0;
		for(let d of data) {
			const v = net.run(d.input)[0] - d.output[0];
			if(p) console.log(d, v + d.output[0]);
			total += v*v;
		}
		return total;
	}

	// train the network
	let l = loss(nn);
	for(let i = 0;i < 1000;i ++) {
		let copy = snn.copyNN(nn);
		snn.mutate(copy, MUTATE_CHANCE);
		const newL = loss(copy);
		if(newL < l) {
			nn = snn.copyNN(copy);
			l = newL;
		}
	}

	return loss(nn) < 0.1;
}

console.log('simple train:', simple_train());
const mc = 0.15;
console.log('xor train: mc =', mc, ',', ((mc) => {
	const total = 1000;
	let ok = 0;
	for(let i = 0;i < total;i ++) {
		ok += xor_train(mc);
	}
	return "" + ok + "/" + total + " = " + ok / total;
})(mc));
