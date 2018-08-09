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

	const finalLoss = loss(nn);
	return {
		passed: finalLoss < 0.01,
		details: 'Loss after all training: ' + finalLoss
	};
	//nn.run([-12, 4, 34, 12]); // can be absouletly anything
}

const xor_train = (MUTATE_CHANCE) => {
	const train = () => {
		let nn = new snn.Network([2, 2, 1], snn.ReLU);
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
		const MUTATE_CHANCE = 0.15;
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

		return {
			passed : l < 0.2, // l holds the lowest loss recorded (the loss of nn after the loop)
			details: 'Loss after all training: ' + l
		};
	}

	const total = 1000;
	let ok = 0;
	for(let i = 0;i < total;i ++) {
		ok += train().passed;
	}

	return {
		passed : (ok / total) > 0.28, // as of 79aa5409005845066a648e885e1a8226577733a2 0.31 is actually the expetced chance (measured after 10000 trains)
		details: ok + '/' + total + ' (' + (ok / total) + ') networks learned to do xor with good enough loss'
	};
}

const performance_run = () => {
	const nn = new snn.Network([1, 100, 1000, 50, 5]);

	const beginTime = new Date();
	for(let i = 0;i < 1000;i ++) {
		nn.run([Math.random()]);
	}
	const timeTaken = new Date() - beginTime;

	return {
		passed : true,
		details: timeTaken + 'ms'
	};
}

console.log('simple train:'       , simple_train());
console.log('xor train:'          , xor_train());
console.log('performance of .run:', performance_run());
