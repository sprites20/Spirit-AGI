---VERTEX SHADER-------------------------------------------------------
#ifdef GL_ES
    precision highp float;
#endif

/* Vertex attributes/inputs, defined in MeshData Object */
attribute vec3 v_pos;
attribute vec3 v_normal;
attribute vec2 v_tc0;

/* Outputs to the fragment shader */
varying vec2 tex_coord0;
varying vec3 normal_vec;  // Use vec3 instead of vec4 for normals
varying vec4 vertex_pos;

uniform mat4 modelview_mat;
uniform mat4 projection_mat;
uniform mat4 normal_mat;

void main(void) {
    vec4 pos = modelview_mat * vec4(v_pos, 1.0);
    vertex_pos = pos;
    normal_vec = normalize(vec3(normal_mat * vec4(v_normal, 0.0)));
    gl_Position = projection_mat * pos;
    tex_coord0 = v_tc0;
}

---FRAGMENT SHADER-----------------------------------------------------
#ifdef GL_ES
    precision highp float;
#endif

/* Outputs from Vertex Shader */
varying vec3 normal_vec;
varying vec4 vertex_pos;
varying vec2 tex_coord0;

uniform sampler2D texture0;

void main(void) {
    vec3 v_light = normalize(-vertex_pos.xyz);  // Assuming light at eye
    float theta = clamp(dot(normalize(normal_vec), v_light), 0.0, 1.0);
    gl_FragColor = vec4(theta, theta, theta, 1.0) * texture2D(texture0, tex_coord0);
}
