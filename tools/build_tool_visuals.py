"""Build display-only meshes of the installed end effector from the reBot CAD context.

The simulation collides with primitive boxes (tools/build_scene.py); for pictures it draws the real
parts: R8 adapter plates, PX6D force/torque sensor, M7 stator, the stock gripper body, the printed
fingertip connectors, DM-Tac W2L tactile fingers and the D405 wrist mount and camera.

    python tools/build_tool_visuals.py <reBot>/hardware/tactile_extraction_fixture/output/setup_20260924

Reads robot_context.json and its STL files (read-only), simplifies each part by vertex clustering,
merges parts per (body, colour) and writes assets/b601/tool_visual/*.stl (metres, already in the frame
of the sim body they hang on) plus manifest.json. Placement follows the website's rig viewer
(github.com/bowenszhu/bowenszhu.github.io, tools/build_rig_model.py): the D405 mount is turned 180 deg
about the gripper axis to the upper side where it is installed, and the M7 stator is split out of the
MJCF link6 mesh, turned to the factory clocking and moved 40 mm out with the gripper.
Requires numpy.
"""
import json
import os
import struct
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'assets', 'b601', 'tool_visual')
MESHES = os.path.join(ROOT, 'assets', 'b601', 'meshes')
SIM_FINGER_BASE = 0.009  # finger body offset along link6 y in tools/build_scene.py
CAD_SLIDE = 0.0114       # W2L pad inner faces at y = +-11.4 mm in the CAD state

# (output name, colour) per CAD part prefix; colours as in the website's rig viewer
PARTS = [
    ('installed_plate_A_R8', 'plate_a', (0.722, 0.533, 0.318)),
    ('installed_plate_B_R8', 'plate_b', (0.427, 0.549, 0.671)),
    ('installed_PX6D', 'px6d', (0.62, 0.64, 0.68)),
    ('installed_M', 'screws', (0.32, 0.34, 0.37)),
    ('pla7_green', 'gripper_green', (0.725, 0.835, 0.118)),
    ('cnc7', 'gripper_cnc', (0.72, 0.76, 0.73)),
    ('motor_7', 'gripper_motor', (0.045, 0.052, 0.049)),
    ('D405_mount', 'd405_mount', (0.8, 0.57, 0.18)),
    ('D405_envelope', 'd405_camera', (0.15, 0.22, 0.27)),
    ('cnc_', 'finger_cnc', (0.72, 0.76, 0.73)),
    ('connector_', 'finger_connector', (0.2, 0.65, 0.8)),
    ('W2L_', 'finger_w2l', (0.76, 0.79, 0.82)),
]


def read_stl(path):
    with open(path, 'rb') as handle:
        data = handle.read()
    count = struct.unpack_from('<I', data, 80)[0]
    if 84 + count * 50 == len(data):
        raw = np.frombuffer(data, dtype=np.dtype([('n', '<f4', 3), ('v', '<f4', (3, 3)), ('a', '<u2')]),
                            count=count, offset=84)
        return raw['v'].astype(np.float64)
    tris, cur = [], []
    for line in data.decode('ascii', 'ignore').splitlines():
        parts = line.split()
        if parts and parts[0] == 'vertex':
            cur.append([float(x) for x in parts[1:4]])
            if len(cur) == 3:
                tris.append(cur)
                cur = []
    return np.array(tris, dtype=np.float64)


def write_stl(path, tris):
    with open(path, 'wb') as handle:
        handle.write(b'\0' * 80 + struct.pack('<I', len(tris)))
        for tri in tris:
            handle.write(struct.pack('<12fH', 0, 0, 0, *tri.ravel(), 0))


def cluster(tris, cell):
    """Vertex-clustering simplification (same as the website's rig builder)."""
    flat = tris.reshape(-1, 3)
    keys = np.floor(flat / cell).astype(np.int64)
    _, inverse = np.unique(keys, axis=0, return_inverse=True)
    inverse = inverse.reshape(-1)
    counts = np.bincount(inverse)
    verts = np.zeros((counts.size, 3))
    for axis in range(3):
        verts[:, axis] = np.bincount(inverse, weights=flat[:, axis]) / counts
    faces = inverse.reshape(-1, 3)
    keep = (faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2]) & (faces[:, 0] != faces[:, 2])
    return verts[faces[keep]]


def components(tris):
    vertices, inverse = np.unique(np.round(tris.reshape(-1, 3), 4), axis=0, return_inverse=True)
    faces = inverse.reshape(-1, 3)
    parents = np.arange(len(vertices))

    def root(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i
    for a, b, c in faces:
        r = root(a)
        parents[root(b)] = r
        parents[root(c)] = r
    labels = np.array([root(i) for i in faces[:, 0]])
    return [tris[labels == lab] for lab in np.unique(labels)]


def transform(tris, m):
    flat = tris.reshape(-1, 3) @ m[:3, :3].T + m[:3, 3]
    return flat.reshape(-1, 3, 3)


def main():
    context_dir = sys.argv[1]
    ctx = json.load(open(os.path.join(context_dir, 'robot_context.json')))
    frames = {k: np.array(v) for k, v in ctx['states'][0]['frames'].items()}
    to_link6 = np.linalg.inv(frames['link6'])
    groups = {}
    for v in ctx['visuals']:
        name = v['name']
        if 'ENVELOPE' in name:
            continue
        spec = next((s for s in PARTS if name.startswith(s[0])), None)
        if spec is None:
            continue
        local = np.array(v['local'])
        if name in ('D405_mount', 'D405_envelope'):
            local = np.diag([1.0, -1.0, -1.0, 1.0]) @ local  # upper side of the gripper
        m = to_link6 @ frames[v['body']] @ local  # mm, link6 frame
        tris = transform(read_stl(os.path.join(context_dir, v['file'])) * float(v['scale']), m) / 1000.0
        side = {'gripper_left': 'left', 'gripper_right': 'right'}.get(v['body'])
        body = 'tool'
        if side:
            tris = tris - np.array([0, (1 if side == 'left' else -1) * (SIM_FINGER_BASE + CAD_SLIDE), 0])
            body = 'finger_' + side
        key = (body, spec[1])
        groups.setdefault(key, {'colour': spec[2], 'tris': []})['tris'].append(tris)
    # M7 stator from the MJCF link6 mesh (the J6 rotor was removed for the sensor stack)
    parts = components(read_stl(os.path.join(MESHES, 'link6.STL')) * 1000)
    stator = next(t for t in parts if np.ptp(t.reshape(-1, 3), axis=0)[0] > 50)
    turn = np.diag([-1.0, -1.0, 1.0, 1.0])
    turn[2, 3] = 40.0
    groups[('tool', 'stator')] = {'colour': (0.60, 0.63, 0.66), 'tris': [transform(stator, turn) / 1000.0]}
    os.makedirs(OUT, exist_ok=True)
    manifest = []
    for (body, name), g in sorted(groups.items()):
        tris = cluster(np.concatenate(g['tris']), 0.0006)
        fn = '%s_%s.stl' % (body, name)
        write_stl(os.path.join(OUT, fn), tris)
        manifest.append({'file': fn, 'body': body, 'rgba': [round(c, 3) for c in g['colour']] + [1]})
        print(fn, len(tris), 'triangles')
    json.dump({'source': 'reBot CAD context setup_20260924 (robot_context.json), state 0',
               'parts': manifest}, open(os.path.join(OUT, 'manifest.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
