#version 330

in vec3 Position;

out block {
	vec3 position;
	vec2 tex_coord;
};

void main()
{
	tex_coord = (Position.xy + vec2(1.0, 1.0)) * 0.5;
	position = Position;
	gl_Position = vec4(Position, 1.0);
}
