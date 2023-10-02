#include "../../../IntNet/intnet.hpp"

#include <AGL/agl.hpp>
#include <cstdlib>
#include <fstream>

#define DEVIATION .5
#define STEPS	  360
#define DISCOUNT  .9

int main()
{
	srand(time(nullptr));

	in::NetworkStructure netStruct(5, {5, 5}, 2, false);

	in::NetworkStructure::randomWeights(netStruct);

	in::NeuralNetwork network(netStruct);

	network.setActivation(in::ActivationFunction::tanh);

	network.setInputNode(0, 1);

	float reward;

	struct
	{
			agl::Vec<float, 2> pos;
			float			   radius = 20;
	} agent;

	agent.pos.x = (rand() / (float)RAND_MAX) * 1920;
	agent.pos.y = (rand() / (float)RAND_MAX) * 1080;

	struct
	{
			agl::Vec<float, 2> pos;
			float			   radius = 10;
	} target;

	target.pos.x = (rand() / (float)RAND_MAX) * 1920;
	target.pos.y = (rand() / (float)RAND_MAX) * 1080;

	struct
	{
			float state[5];
			float action[2];
			float logProb; // squared difference between action and taken action summed up
			float reward;
			float retRaw;
	} trajectory[STEPS];

	agl::RenderWindow window;
	window.setup({1920, 1080}, "SpaceFly");
	window.setClearColor(agl::Color::Black);
	window.setFPS(0);

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

	int steps = 0;

	while (!event.windowClose())
	{
		event.poll();

		window.clear();

		// draw agent
		circle.setColor(agl::Color::Cyan);
		circle.setSize({agent.radius, agent.radius});
		circle.setPosition({agent.pos});
		window.drawShape(circle);

		target.pos = event.getPointerWindowPosition();

		// draw target
		circle.setColor(agl::Color::Magenta);
		circle.setSize({target.radius, target.radius});
		circle.setPosition({target.pos});
		window.drawShape(circle);

		window.display();

		// network shit

		network.setInputNode(1, agent.pos.x / 1920);
		network.setInputNode(2, agent.pos.x / 1080);
		network.setInputNode(3, target.pos.x / 1920);
		network.setInputNode(4, target.pos.x / 1080);

		network.update();

		float xAction = network.outputNode[0].value * 2;
		float yAction = network.outputNode[1].value * 2;

		float range;

		range	= ((rand() / (float)RAND_MAX) * 2) - 1;
		xAction = xAction + (DEVIATION * range);
		range	= ((rand() / (float)RAND_MAX) * 2) - 1;
		yAction = yAction + (DEVIATION * range);

		xAction = fmin(xAction, 1);
		xAction = fmax(xAction, -1);
		yAction = fmin(yAction, 1);
		yAction = fmax(yAction, -1);

		agent.pos += {xAction, yAction};

		reward += 1 / (agent.pos - target.pos).length();
		reward *= 1000;

		trajectory[steps].state[0] = network.inputNode[0]->value;
		trajectory[steps].state[1] = network.inputNode[1]->value;
		trajectory[steps].state[2] = network.inputNode[2]->value;
		trajectory[steps].state[3] = network.inputNode[3]->value;
		trajectory[steps].state[4] = network.inputNode[4]->value;

		trajectory[steps].action[0] = xAction;
		trajectory[steps].action[1] = yAction;

		for (int i = 0; i < 2; i++)
		{
			trajectory[steps].logProb += pow(network.outputNode[i].value - trajectory[steps].action[i], 2);
		}

		trajectory[steps].logProb *=- -.5;

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
			trajectory[i].retRaw += trajectory[i].reward * std::pow(DISCOUNT, x - i);
		}

		loss += trajectory[i].retRaw * trajectory[i].logProb;
	}

	window.close();

	network.destroy();
}
