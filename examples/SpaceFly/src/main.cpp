#include "../../../IntNet/intnet.hpp"

#include <AGL/agl.hpp>
#include <cstdlib>
#include <fstream>

#define DEVIATION .5
#define STEPS	  360
#define DISCOUNT  .9

struct Tracjectory
{
		float state[3];
		float action[2];
		float logProb; // squared difference between action and taken action summed up
		float reward;
		float retRaw;
};

float noisyPick(float val, float variation)
{
	float range = ((rand() / (float)RAND_MAX) * 2) - 1;
	return val + (variation * range);
}

int main()
{
	srand(0);

	in::NetworkStructure netStruct(3, {}, 2, false);

	in::NetworkStructure::randomWeights(netStruct);

	in::NeuralNetwork network(netStruct);

	network.setActivation(in::ActivationFunction::tanh);

	network.setInputNode(0, 1);

	agl::RenderWindow window;
	window.setup({1920, 1080}, "SpaceFly");
	window.setClearColor(agl::Color::Black);
	window.setFPS(0);

	window.setSwapInterval(0);

	agl::Event event;
	event.setWindow(window);

	agl::Camera camera;
	camera.setView({0, 0, 50}, {0, 0, 0}, {0, 1, 0});
	camera.setOrthographicProjection(0, 1920, 1080, 0, 0.1, 100);

	agl::Texture blank;
	blank.setBlank();

	agl::ShaderBuilder sbv;
	sbv.setDefaultVert();

	agl::ShaderBuilder sbf;
	sbf.setDefaultFrag();

	agl::Shader shader;

	{
		std::string s1 = sbv.getSrc();
		std::string s2 = sbf.getSrc();

		shader.compileSrc(s1, s2);
	}

	shader.use();

	window.getShaderUniforms(shader);
	window.updateMvp(camera);

	agl::Circle circle(12);
	circle.setTexture(&blank);
	circle.setColor(agl::Color::White);
	circle.setSize({100, 100});
	circle.setPosition({100, 100});

	network.learningRate = .1;

	while (!event.windowClose())
	{
		// std::cout << network.structure << '\n';

		int steps = 0;

		struct
		{
				agl::Vec<float, 2> pos;
				float			   radius = 20;
		} agent;

		struct
		{
				agl::Vec<float, 2> pos;
				float			   radius = 10;
		} target;

		int r = rand();

		agent.pos.x = (rand() / (float)RAND_MAX) * 1920;
		agent.pos.y = (rand() / (float)RAND_MAX) * 1080;

		target.pos.x = (rand() / (float)RAND_MAX) * 1920;
		target.pos.y = (rand() / (float)RAND_MAX) * 1080;

		Tracjectory trajectory[STEPS];
		float		reward;

		network.learningRate = .1;

		while (!event.windowClose())
		{
			if (event.isKeyPressed(agl::Key::Space))
			{
				window.setSwapInterval(0);
			}
			else
			{
				window.setSwapInterval(1);
			}

			event.poll();

			window.clear();

			// draw agent
			circle.setColor(agl::Color::Cyan);
			circle.setSize({agent.radius, agent.radius});
			circle.setPosition({agent.pos});
			window.drawShape(circle);

			// target.pos = event.getPointerWindowPosition();

			// draw target
			circle.setColor(agl::Color::Magenta);
			circle.setSize({target.radius, target.radius});
			circle.setPosition({target.pos});
			window.drawShape(circle);

			window.display();

			// network shit

			network.setInputNode(0, 1);
			network.setInputNode(1, (target.pos.x - agent.pos.x) / 1920);
			network.setInputNode(2, (target.pos.y - agent.pos.y) / 1080);

			network.update();

			float xAction = network.outputNode[0].value;
			float yAction = network.outputNode[1].value;

			float range;

			xAction = noisyPick(xAction, DEVIATION);
			yAction = noisyPick(yAction, DEVIATION);

			xAction = fmin(xAction, 1);
			xAction = fmax(xAction, -1);
			yAction = fmin(yAction, 1);
			yAction = fmax(yAction, -1);

			float beforeDist = (agent.pos - target.pos).length();

			agent.pos += {xAction, yAction};

			float afterDist = (agent.pos - target.pos).length();

			reward = beforeDist - afterDist;

			trajectory[steps].state[0] = network.inputNode[0]->value;
			trajectory[steps].state[1] = network.inputNode[1]->value;
			trajectory[steps].state[2] = network.inputNode[2]->value;

			trajectory[steps].action[0] = xAction;
			trajectory[steps].action[1] = yAction;

			for (int i = 0; i < 2; i++)
			{
				trajectory[steps].logProb += pow(network.outputNode[i].value - trajectory[steps].action[i], 2);
			}

			trajectory[steps].logProb *= -.5;

			// std::cout << trajectory[steps].logProb << '\n';

			trajectory[steps].reward = reward;

			reward = 0;

			steps++;
			if (steps >= STEPS)
			{
				break;
			}
		}

		float loss = 0;

		for (int i = 0; i < STEPS; i++)
		{
			trajectory[i].retRaw = trajectory[i].reward;
			for (int x = i + 1; x < STEPS; x++)
			{
				trajectory[i].retRaw += trajectory[x].reward * std::pow(DISCOUNT, x - i);
			}

			// loss += trajectory[i].retRaw * trajectory[i].logProb;
			loss += trajectory[i].retRaw;
		}

		loss /= STEPS;

		std::cout << loss << '\n';

		std::vector<float> gradients;

		network.setupGradients(&gradients);

		for (int i = 0; i < STEPS; i++)
		{
			network.setInputNode(0, trajectory[i].state[0]);
			network.setInputNode(1, trajectory[i].state[1]);
			network.setInputNode(2, trajectory[i].state[2]);

			network.update();

			std::vector<float> target = {trajectory[i].action[0], trajectory[i].action[1]};

			network.calcGradients(&gradients, target);
		}

		// for (float &g : gradients)
		// {
		// 	g = noisyPick(g, .1);
		// }

		network.applyGradients(gradients, loss, STEPS);

		// std::cout << network.structure << '\n';
		// std::cout << loss << '\n';
	}

	window.close();

	network.destroy();
}
