class MeshData(object):
    def __init__(self, **kwargs):
        self.name = kwargs.get("name")
        self.vertex_format = [
            (b'v_pos', 3, 'float'),
            (b'v_normal', 3, 'float'),
            (b'v_tc0', 2, 'float')]
        self.vertices = []
        self.indices = []

    def calculate_normals(self):
        for i in range(0, len(self.indices), 3):
            v1_idx, v2_idx, v3_idx = self.indices[i], self.indices[i + 1], self.indices[i + 2]
            v1, v2, v3 = self.vertices[v1_idx], self.vertices[v2_idx], self.vertices[v3_idx]

            # Calculate the normal for this face
            edge1 = [v2[j] - v1[j] for j in range(3)]
            edge2 = [v3[j] - v1[j] for j in range(3)]
            normal = [
                edge1[1] * edge2[2] - edge1[2] * edge2[1],
                edge1[2] * edge2[0] - edge1[0] * edge2[2],
                edge1[0] * edge2[1] - edge1[1] * edge2[0]
            ]

            # Assign the normal to the vertices
            for j in range(3):
                self.vertices[v1_idx + j + 3] = normal
                self.vertices[v2_idx + j + 3] = normal
                self.vertices[v3_idx + j + 3] = normal



class ObjFile:
    def finish_object(self):
        if self._current_object is None:
            return
    
        mesh = MeshData()
        idx = 0
        for f in self.faces:
            verts = f[0]
            norms = f[1]
            tcs = f[2]
            for i in range(3):
                # get normal components
                n = (0.0, 0.0, 0.0)
                if norms[i] != -1:
                    n = self.normals[norms[i] - 1]

                # get texture coordinate components
                t = (0.0, 0.0)
                if tcs[i] != -1:
                    t = self.texcoords[tcs[i] - 1]

                # get vertex components
                v = self.vertices[verts[i] - 1]

                data = [v[0], v[1], v[2], n[0], n[1], n[2], t[0], t[1]]
                mesh.vertices.extend(data)

            tri = [idx, idx + 1, idx + 2]
            mesh.indices.extend(tri)
            idx += 3

        self.objects[self._current_object] = mesh
        # mesh.calculate_normals()
        self.faces = []

    def __init__(self, filename, swapyz=False):
        """Loads a Wavefront OBJ file. """
        self.objects = {}
        self.vertices = []
        self.normals = []
        self.texcoords = []
        self.faces = []

        self._current_object = None

        material = None
        for line in open(filename, "r"):
            if line.startswith('#'):
                continue
            if line.startswith('s'):
                continue
            values = line.split()
            if not values:
                continue
            if values[0] == 'o':
                self.finish_object()
                self._current_object = values[1]
            elif values[0] == 'mtllib':
                self.mtl = MTL(values[1])
            elif values[0] in ('usemtl', 'usemat'):
                if len(values) > 1:
                    material = values[1]
                else:
                    # Handle the case when there is no material name
                    material = None  # Or some default value
            if values[0] == 'v':
                v = list(map(float, values[1:4]))
                if swapyz:
                    v = v[0], v[2], v[1]
                self.vertices.append(v)
            elif values[0] == 'vn':
                v = list(map(float, values[1:4]))
                if swapyz:
                    v = v[0], v[2], v[1]
                self.normals.append(v)
            elif values[0] == 'vt':
                self.texcoords.append(list(map(float, values[1:3])))
            elif values[0] == 'f':
                face = []
                texcoords = []
                norms = []
                for v in values[1:]:
                    w = v.split('/')
                    face.append(int(w[0]))
                    if len(w) >= 2 and len(w[1]) > 0:
                        texcoords.append(int(w[1]))
                    else:
                        texcoords.append(-1)
                    if len(w) >= 3 and len(w[2]) > 0:
                        norms.append(int(w[2]))
                    else:
                        norms.append(-1)
                self.faces.append((face, norms, texcoords, material))
                
                
        self.finish_object()


def MTL(filename):
    contents = {}
    mtl = None
    return
    for line in open(filename, "r"):
        if line.startswith('#'):
            continue
        values = line.split()
        if not values:
            continue
        if values[0] == 'newmtl':
            mtl = contents[values[1]] = {}
        elif mtl is None:
            raise ValueError("mtl file doesn't start with newmtl stmt")
        mtl[values[0]] = values[1:]
    return contents