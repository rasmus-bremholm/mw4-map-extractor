#!/usr/bin/env python3
"""Export a CoD4 (IW3) map from your own install to a Blender-loadable glTF scene.

    python3 export_map.py "<Call of Duty 4 folder>" mp_crash "<output folder>"

Writes  <out>/<map>.gltf + <map>.bin  (world mesh, every prop model placed where the
map puts it) and <out>/textures/*.png (colour maps, converted from the game's IWDs).
Units are left as CoD units (1 unit = 1 inch, Z up); the root node carries the
scale/rotation that turns that into metres / glTF axes.
"""
import sys, os, re, json, math, struct, zipfile, io, argparse, urllib.parse, collections
import numpy as np
from iw3zone import Spec, Zone, load_ff

HERE = os.path.dirname(os.path.abspath(__file__))

WV = np.dtype([('xyz', '<f4', 3), ('bs', '<f4'), ('color', 'u1', 4), ('uv', '<f4', 2), ('lm', '<f4', 2),
               ('n', 'u1', 4), ('t', 'u1', 4)])
PV = np.dtype([('xyz', '<f4', 3), ('bs', '<f4'), ('color', 'u1', 4), ('uv', '<u2', 2), ('n', 'u1', 4),
               ('t', 'u1', 4)])
assert WV.itemsize == 44 and PV.itemsize == 32


def unpack_normal(n):
    n = n.astype(np.float32)
    scale = (n[:, 3:4] + 192.0) / 32385.0
    v = (n[:, :3] - 127.0) * scale
    l = np.linalg.norm(v, axis=1, keepdims=True)
    l[l == 0] = 1
    return (v / l).astype(np.float32)


def half_to_float(h):
    h = h.astype(np.uint32)
    a = (h << np.uint32(14)) & np.uint32(0x0FFFC000)
    b = (~(h << np.uint32(14))) & np.uint32(0x10000000)
    bits = ((h << np.uint32(16)) & np.uint32(0x80000000)) | (((a - b) ^ np.uint32(0x80000000)) >> np.uint32(1))
    f = bits.view(np.float32).copy()
    f[h == 0] = 0.0
    return f


class LoadedZone:
    def __init__(self, spec, path):
        self.name = os.path.basename(path)
        self.z = Zone(spec, load_ff(path))
        self.z.run()
        z = self.z
        self.by = collections.defaultdict(dict)
        for tn, a, tid in z.assets:
            nm = self.asset_name(tn, a)
            if nm is not None:
                self.by[tn].setdefault(nm, a)

    def asset_name(self, tn, a):
        z = self.z
        path = {'Material': 'info::name', 'snd_alias_list_t': 'aliasName', 'Font_s': 'fontName',
                'menuDef_t': 'window::name', 'WeaponDef': 'szInternalName', 'SndCurve': 'filename'}.get(tn, 'name')
        try:
            return z.string(tn, a, path)
        except Exception:
            return None


class Exporter:
    def __init__(self, game_dir, mapname, out_dir, fallbacks=('common_mp',), lang='english'):
        self.game = game_dir; self.mapname = mapname; self.out = out_dir
        self.spec = Spec(os.path.join(HERE, 'iw3types.json'), os.path.join(HERE, 'zonecmds'))
        self.zone_dir = os.path.join(game_dir, 'zone', lang)
        self.main = LoadedZone(self.spec, os.path.join(self.zone_dir, mapname + '.ff'))
        self.fallback_names = list(fallbacks); self.fb = {}
        # gltf state
        self.bin = bytearray(); self.views = []; self.accessors = []
        self.meshes = []; self.nodes = []; self.materials = []; self.textures = []; self.images = []
        self.mat_index = {}; self.img_index = {}; self.mesh_index = {}
        self.matinfo = {}
        self.missing_models = collections.Counter()

    # ---------- zone lookups (with fallback zones for ",name" stub assets)
    def fallback(self, i):
        n = self.fallback_names[i]
        if n not in self.fb:
            p = os.path.join(self.zone_dir, n + '.ff')
            self.fb[n] = LoadedZone(self.spec, p) if os.path.exists(p) else None
        return self.fb[n]

    def find_asset(self, tn, name):
        """-> (LoadedZone, addr) of a real (non-stub) asset, or None"""
        name = name.lstrip(',')
        a = self.main.by[tn].get(name)
        if a is not None:
            return self.main, a
        for i in range(len(self.fallback_names)):
            lz = self.fallback(i)
            if lz is None:
                continue
            a = lz.by[tn].get(name)
            if a is not None:
                return lz, a
        return None

    # ---------- gltf buffer helpers
    def add_view(self, data, target=None):
        while len(self.bin) % 4:
            self.bin.append(0)
        off = len(self.bin)
        self.bin += data
        v = {'buffer': 0, 'byteOffset': off, 'byteLength': len(data)}
        if target:
            v['target'] = target
        self.views.append(v)
        return len(self.views) - 1

    def add_acc(self, arr, kind, target=None, minmax=False):
        arr = np.ascontiguousarray(arr)
        ct = {np.dtype('<f4'): 5126, np.dtype('<u4'): 5125, np.dtype('<u2'): 5123}[arr.dtype]
        a = {'bufferView': self.add_view(arr.tobytes(), target), 'componentType': ct, 'count': int(arr.shape[0]),
             'type': kind}
        if minmax:
            a['min'] = [float(x) for x in arr.min(axis=0)]
            a['max'] = [float(x) for x in arr.max(axis=0)]
        self.accessors.append(a)
        return len(self.accessors) - 1

    def add_prim(self, pos, nrm, uv, tris, material):
        """tris: (n,3) uint32 indexing pos/nrm/uv. Compacts to the used vertices."""
        tris = np.asarray(tris, dtype=np.uint32).reshape(-1, 3)
        if len(tris) == 0:
            return None
        used, inv = np.unique(tris, return_inverse=True)
        idx = inv.astype(np.uint32).reshape(-1, 3)
        idx = idx[:, [0, 2, 1]]  # game winding is clockwise, glTF wants counter-clockwise
        p = {'attributes': {'POSITION': self.add_acc(pos[used].astype('<f4'), 'VEC3', 34962, True),
                            'NORMAL': self.add_acc(nrm[used].astype('<f4'), 'VEC3', 34962),
                            'TEXCOORD_0': self.add_acc(uv[used].astype('<f4'), 'VEC2', 34962)},
             'indices': self.add_acc(idx.reshape(-1), 'SCALAR', 34963), 'mode': 4}
        if material is not None:
            p['material'] = material
        return p

    # ---------- materials
    def material(self, lz, addr):
        z = lz.z
        name = z.string('Material', addr, 'info::name') or 'unnamed'
        if name.startswith(','):
            r = self.find_asset('Material', name)
            if r is None:
                return self.stub_material(name.lstrip(','))
            lz, addr = r; z = lz.z; name = name.lstrip(',')
        if name in self.mat_index:
            return self.mat_index[name]
        E = self.spec.enums
        tex = {}
        tt = z.follow('Material', addr, 'textureTable')
        if tt and tt[0] in ('data', 'alias') and tt[2] is not None:
            n = z.get('Material', addr, 'textureCount')
            for k in range(n):
                a = tt[2] + 12 * k
                sem = z.d[a + 7]
                img = z.ptr.get(a + 8)
                if sem == E['TS_WATER_MAP'] or not img or img[0] != 'asset' or img[2] is None:
                    continue
                iname = z.string('GfxImage', img[2], 'name')
                if iname:
                    tex.setdefault(sem, iname.lstrip(','))
        color = tex.get(E['TS_COLOR_MAP'])
        # alpha / culling from state bits
        alpha = 'OPAQUE'; twosided = False
        sb = z.follow('Material', addr, 'stateBitsTable')
        if sb and sb[2] is not None:
            n = z.get('Material', addr, 'stateBitsCount')
            for k in range(n):
                b0, b1 = struct.unpack_from('<II', z.d, sb[2] + 8 * k)
                if not (b0 & E['GFXS0_ATEST_DISABLE']) and alpha == 'OPAQUE':
                    alpha = 'MASK'
                src = (b0 & E['GFXS0_SRCBLEND_RGB_MASK']) >> E['GFXS0_SRCBLEND_RGB_SHIFT']
                dst = (b0 & E['GFXS0_DSTBLEND_RGB_MASK']) >> E['GFXS0_DSTBLEND_RGB_SHIFT']
                op = (b0 & E['GFXS0_BLENDOP_RGB_MASK']) >> E['GFXS0_BLENDOP_RGB_SHIFT']
                if k == 0 and op != E['GFXS_BLENDOP_DISABLED'] and src == E['GFXS_BLEND_SRCALPHA'] \
                        and dst == E['GFXS_BLEND_INVSRCALPHA']:
                    alpha = 'BLEND'
                if 'GFXS0_CULL_MASK' in E and (b0 & E['GFXS0_CULL_MASK']) == E.get('GFXS0_CULL_NONE', -1):
                    twosided = True
        ts = z.follow('Material', addr, 'techniqueSet')
        techset = z.string('MaterialTechniqueSet', ts[2], 'name') if ts and ts[0] == 'asset' and ts[2] is not None else None
        m = {'name': name, 'pbrMetallicRoughness': {'metallicFactor': 0.0, 'roughnessFactor': 0.9}}
        if alpha != 'OPAQUE':
            m['alphaMode'] = alpha
            if alpha == 'MASK':
                m['alphaCutoff'] = 0.5
        if twosided:
            m['doubleSided'] = True
        self.materials.append(m)
        i = self.mat_index[name] = len(self.materials) - 1
        self.matinfo[name] = {'index': i, 'colorMap': color, 'normalMap': tex.get(E['TS_NORMAL_MAP']),
                              'specularMap': tex.get(E['TS_SPECULAR_MAP']), 'alphaMode': alpha,
                              'techniqueSet': (techset or '').lstrip(','), 'zone': lz.name,
                              'surfaceTypeBits': z.get('Material', addr, 'info::surfaceTypeBits')}
        return i

    def stub_material(self, name):
        if name not in self.mat_index:
            self.materials.append({'name': name, 'pbrMetallicRoughness': {'baseColorFactor': [1, 0, 1, 1]}})
            self.mat_index[name] = len(self.materials) - 1
            self.matinfo[name] = {'index': self.mat_index[name], 'colorMap': None, 'missing': True}
        return self.mat_index[name]

    # ---------- xmodels
    def xmodel_mesh(self, name):
        name = name.lstrip(',')
        if name in self.mesh_index:
            return self.mesh_index[name]
        r = self.find_asset('XModel', name)
        mi = None
        if r is not None:
            lz, a = r; z = lz.z
            numsurfs, surfidx = struct.unpack_from('<HH', z.d, z.field('XModel', a, 'lodInfo')[0] + 4)
            surfs = z.follow('XModel', a, 'surfs'); mh = z.follow('XModel', a, 'materialHandles')
            prims = []
            if surfs and surfs[2] is not None and mh and mh[2] is not None:
                ssz = self.spec.structs['XSurface']['size']
                for k in range(surfidx, surfidx + numsurfs):
                    sa = surfs[2] + k * ssz
                    vc = z.get('XSurface', sa, 'vertCount'); tc = z.get('XSurface', sa, 'triCount')
                    v0 = z.follow('XSurface', sa, 'verts0'); ti = z.follow('XSurface', sa, 'triIndices')
                    if not v0 or not ti or v0[2] is None or ti[2] is None or vc == 0 or tc == 0:
                        continue
                    v = np.frombuffer(z.d, PV, vc, v0[2])
                    tris = np.frombuffer(z.d, '<u2', tc * 3, ti[2]).astype(np.uint32).reshape(-1, 3)
                    uv = np.stack([half_to_float(v['uv'][:, 1]), half_to_float(v['uv'][:, 0])], axis=1)
                    mp = z.ptr.get(mh[2] + 4 * k)
                    mat = self.material(lz, mp[2]) if mp and mp[0] == 'asset' and mp[2] is not None else None
                    p = self.add_prim(v['xyz'], unpack_normal(v['n']), uv, tris, mat)
                    if p:
                        prims.append(p)
            if prims:
                self.meshes.append({'name': name, 'primitives': prims})
                mi = len(self.meshes) - 1
        self.mesh_index[name] = mi
        return mi

    def add_node(self, node, parent):
        self.nodes.append(node)
        i = len(self.nodes) - 1
        self.nodes[parent].setdefault('children', []).append(i)
        return i

    def add_instance(self, parent, model, origin, axis, scale, counter):
        mi = self.xmodel_mesh(model)
        if mi is None:
            self.missing_models[model.lstrip(',')] += 1
            return
        name = model.lstrip(',')
        counter[name] += 1
        o = origin; ax = axis
        m = [ax[0][0] * scale, ax[0][1] * scale, ax[0][2] * scale, 0,
             ax[1][0] * scale, ax[1][1] * scale, ax[1][2] * scale, 0,
             ax[2][0] * scale, ax[2][1] * scale, ax[2][2] * scale, 0,
             o[0], o[1], o[2], 1]
        self.add_node({'name': '%s.%03d' % (name, counter[name]), 'mesh': mi, 'matrix': [float(x) for x in m]}, parent)

    # ---------- the map
    def run(self, textures=True, extra_maps=False):
        lz = self.main; z = lz.z; d = z.d
        s = 0.0254
        self.nodes.append({'name': self.mapname,
                           'matrix': [s, 0, 0, 0, 0, 0, -s, 0, 0, s, 0, 0, 0, 0, 0, 1]})
        root = 0
        gw = [a for a in z.assets if a[0] == 'GfxWorld'][0][1]
        G = lambda p, fmt=None: z.get('GfxWorld', gw, p, fmt)
        vd = z.follow('GfxWorld', gw, 'vd::vertices'); ix = z.follow('GfxWorld', gw, 'indices')
        sf = z.follow('GfxWorld', gw, 'dpvs::surfaces')
        V = np.frombuffer(d, WV, vd[3], vd[2])
        I = np.frombuffer(d, '<u2', ix[3], ix[2]).astype(np.uint32)
        pos = V['xyz']; nrm = unpack_normal(V['n']); uv = V['uv']
        nsurf = sf[3]
        S = np.frombuffer(d, np.dtype([('layer', '<i4'), ('first', '<i4'), ('vc', '<u2'), ('tc', '<u2'),
                                       ('base', '<i4'), ('mat', '<u4'), ('misc', 'u1', 4), ('b', '<f4', 6)]),
                          nsurf, sf[2])
        surf_mat = []
        for k in range(nsurf):
            mp = z.ptr.get(sf[2] + 48 * k + 16)
            surf_mat.append(self.material(lz, mp[2]) if mp and mp[0] == 'asset' and mp[2] is not None else None)

        def surf_tris(k):
            b = int(S['base'][k]); n = int(S['tc'][k]) * 3
            return I[b:b + n] + np.uint32(S['first'][k])

        # which surfaces belong to brush models (doors, script brushes...) other than the world itself
        bm = z.follow('GfxWorld', gw, 'models'); nbm = G('modelCount')
        owner = np.zeros(nsurf, dtype=np.int32)
        bms = []
        for k in range(nbm):
            a = bm[2] + 56 * k
            mins = struct.unpack_from('<3f', d, a); maxs = struct.unpack_from('<3f', d, a + 12)
            cnt, start, cnt_nd = struct.unpack_from('<HHH', d, a + 48)
            bms.append((mins, maxs, cnt, start))
            if k > 0:
                owner[start:start + cnt] = k

        def mesh_for(surfs, name):
            groups = collections.defaultdict(list)
            for k in surfs:
                groups[surf_mat[k]].append(surf_tris(k))
            prims = []
            for mat, lst in sorted(groups.items(), key=lambda kv: -1 if kv[0] is None else kv[0]):
                p = self.add_prim(pos, nrm, uv, np.concatenate(lst), mat)
                if p:
                    prims.append(p)
            if not prims:
                return None
            self.meshes.append({'name': name, 'primitives': prims})
            return len(self.meshes) - 1

        def is_sky(k):
            m = surf_mat[k]
            return m is not None and 'sky' in self.materials[m]['name'].split('/')[-1][:4]
        world = [k for k in range(nsurf) if owner[k] == 0 and not is_sky(k)]
        sky = [k for k in range(nsurf) if owner[k] == 0 and is_sky(k)]
        mi = mesh_for(world, 'world')
        if mi is not None:
            self.add_node({'name': 'world', 'mesh': mi}, root)
        mi = mesh_for(sky, 'world_sky')
        if mi is not None:
            self.add_node({'name': 'world_sky', 'mesh': mi}, root)

        # entities
        ents = []
        cm = [a for a in z.assets if a[0] == 'clipMap_t']
        me = [a for a in z.assets if a[0] == 'MapEnts']
        if me:
            es = z.follow('MapEnts', me[0][1], 'entityString')
            txt = d[es[2]:es[2] + es[3]].decode('latin1').rstrip('\0')
            ents = [dict(re.findall(r'"([^"]*)" "([^"]*)"', e)) for e in re.findall(r'\{(.*?)\}', txt, re.S)]
            with open(os.path.join(self.out, self.mapname + '_entities.txt'), 'w') as f:
                f.write(txt)
        bm_origin = {}
        for e in ents:
            m = e.get('model', '')
            if m.startswith('*'):
                bm_origin[int(m[1:])] = [float(x) for x in e.get('origin', '0 0 0').split()]
        if nbm > 1:
            grp = self.add_node({'name': 'brushmodels'}, root)
            for k in range(1, nbm):
                mi = mesh_for([i for i in range(nsurf) if owner[i] == k], 'brushmodel_%d' % k)
                if mi is None:
                    continue
                n = {'name': 'brushmodel_%d' % k, 'mesh': mi}
                if k in bm_origin and self.bm_local:
                    n['translation'] = bm_origin[k]
                self.add_node(n, grp)

        # static models
        cnt = collections.Counter()
        si = z.follow('GfxWorld', gw, 'dpvs::smodelDrawInsts')
        if si and si[2] is not None:
            grp = self.add_node({'name': 'static_models'}, root)
            for k in range(si[3]):
                a = si[2] + 76 * k
                f = struct.unpack_from('<13f', d, a + 4)
                mp = z.ptr.get(a + 56)
                if not mp or mp[2] is None:
                    continue
                name = z.string('XModel', mp[2], 'name')
                self.add_instance(grp, name, f[0:3], (f[3:6], f[6:9], f[9:12]), f[12], cnt)
        # dynamic entities (physics clutter / destructibles)
        if cm:
            ca = cm[0][1]
            ndyn = z.get('clipMap_t', ca, 'dynEntCount', '2H')
            addr, fld = z.field('clipMap_t', ca, 'dynEntDefList')
            dl = z.ptr.get(addr)
            dsz = self.spec.structs['DynEntityDef']['size']
            if dl and dl[2] is not None and ndyn[0]:
                grp = self.add_node({'name': 'dyn_ents'}, root)
                xo = self.spec.structs['DynEntityDef']['byname']['xModel']['off']
                for k in range(ndyn[0]):
                    a = dl[2] + k * dsz
                    q = struct.unpack_from('<4f', d, a + 4); o = struct.unpack_from('<3f', d, a + 20)
                    mp = z.ptr.get(a + xo)
                    if not mp or mp[2] is None:
                        continue
                    name = z.string('XModel', mp[2], 'name')
                    self.add_instance(grp, name, o, quat_axis(q), 1.0, cnt)
        # script models from the entity list
        sm = [e for e in ents if e.get('model') and not e['model'].startswith('*') and 'origin' in e]
        if sm:
            grp = self.add_node({'name': 'script_models'}, root)
            for e in sm:
                o = [float(x) for x in e['origin'].split()]
                ang = [float(x) for x in e.get('angles', '0 0 0').split()]
                self.add_instance(grp, e['model'], o, angles_axis(ang), float(e.get('modelscale', 1) or 1), cnt)
        self.instance_counts = cnt
        self.world_bounds = (G('mins', '3f'), G('maxs', '3f'))
        self.stats = {'world_tris': int(sum(int(S['tc'][k]) for k in world)), 'world_verts': int(len(V)),
                      'surfaces': int(nsurf), 'brushmodels': int(nbm - 1)}

    bm_local = True

    # ---------- textures
    def index_iwds(self):
        idx = {}
        main = os.path.join(self.game, 'main')
        files = sorted(f for f in os.listdir(main) if f.lower().endswith('.iwd'))
        self.zips = {}
        for f in files:
            zf = zipfile.ZipFile(os.path.join(main, f))
            self.zips[f] = zf
            for n in zf.namelist():
                ln = n.lower()
                if ln.startswith('images/') and ln.endswith('.iwi'):
                    idx[ln[7:-4]] = (f, n)
        return idx

    def convert_textures(self, kinds=('colorMap',), limit_seconds=None, log=print):
        import time
        from PIL import Image
        t0 = time.time()
        tdir = os.path.join(self.out, 'textures')
        os.makedirs(tdir, exist_ok=True)
        idx = self.index_iwds()
        wanted = collections.OrderedDict()
        for mname, mi in self.matinfo.items():
            for kind in kinds:
                if mi.get(kind):
                    wanted[mi[kind]] = None
        done = 0; missing = []; pending = 0
        info = {}
        for iname in wanted:
            fn = safe_name(iname) + '.png'
            path = os.path.join(tdir, fn)
            if os.path.exists(path):
                info[iname] = fn; continue
            if limit_seconds and time.time() - t0 > limit_seconds:
                pending += 1; continue
            hit = idx.get(iname.lower())
            if not hit:
                missing.append(iname); continue
            try:
                im = iwi_to_image(self.zips[hit[0]].read(hit[1]))
            except Exception as ex:
                missing.append('%s (%s)' % (iname, ex)); continue
            im.save(path)
            info[iname] = fn; done += 1
        self.tex_files = info
        log('textures: %d converted now, %d available, %d missing, %d still pending' % (done, len(info), len(missing), pending))
        return missing, pending

    def link_textures(self):
        from PIL import Image
        tdir = os.path.join(self.out, 'textures')
        for mname, mi in self.matinfo.items():
            cm = mi.get('colorMap')
            fn = self.tex_files.get(cm) if cm else None
            if not fn:
                continue
            if cm not in self.img_index:
                self.images.append({'uri': 'textures/' + urllib.parse.quote(fn), 'name': cm})
                self.textures.append({'source': len(self.images) - 1, 'sampler': 0})
                self.img_index[cm] = len(self.textures) - 1
            m = self.materials[mi['index']]
            m['pbrMetallicRoughness']['baseColorTexture'] = {'index': self.img_index[cm]}
            mi['colorMapFile'] = 'textures/' + fn

    def write(self):
        while len(self.bin) % 4:
            self.bin.append(0)
        g = {'asset': {'version': '2.0', 'generator': 'cod4 map export'},
             'scene': 0, 'scenes': [{'name': self.mapname, 'nodes': [0]}],
             'nodes': self.nodes, 'meshes': self.meshes, 'materials': self.materials,
             'accessors': self.accessors, 'bufferViews': self.views,
             'buffers': [{'uri': self.mapname + '.bin', 'byteLength': len(self.bin)}]}
        if self.textures:
            g['textures'] = self.textures; g['images'] = self.images
            g['samplers'] = [{'magFilter': 9729, 'minFilter': 9987, 'wrapS': 10497, 'wrapT': 10497}]
        with open(os.path.join(self.out, self.mapname + '.bin'), 'wb') as f:
            f.write(self.bin)
        with open(os.path.join(self.out, self.mapname + '.gltf'), 'w') as f:
            json.dump(g, f)
        with open(os.path.join(self.out, self.mapname + '_materials.json'), 'w') as f:
            json.dump(self.matinfo, f, indent=1, sort_keys=True)


def quat_axis(q):
    x, y, z, w = q
    return ((1 - 2 * (y * y + z * z), 2 * (x * y + z * w), 2 * (x * z - y * w)),
            (2 * (x * y - z * w), 1 - 2 * (x * x + z * z), 2 * (y * z + x * w)),
            (2 * (x * z + y * w), 2 * (y * z - x * w), 1 - 2 * (x * x + y * y)))


def angles_axis(a):
    p, y, r = (math.radians(v) for v in (a + [0, 0, 0])[:3])
    sp, cp, sy, cy, sr, cr = math.sin(p), math.cos(p), math.sin(y), math.cos(y), math.sin(r), math.cos(r)
    fwd = (cp * cy, cp * sy, -sp)
    right = (-sr * sp * cy + cr * sy, -sr * sp * sy - cr * cy, -sr * cp)
    up = (cr * sp * cy + sr * sy, cr * sp * sy - sr * cy, cr * cp)
    return (fwd, tuple(-v for v in right), up)


def safe_name(n):
    return re.sub(r'[^A-Za-z0-9_.\-]', '_', n)


def iwi_to_image(data):
    from PIL import Image
    if data[:3] != b'IWi':
        raise ValueError('not an IWI')
    ver = data[3]
    if ver not in (5, 6):
        raise ValueError('IWI version %d' % ver)
    fmt, flags, w, h, depth = struct.unpack_from('<BBHHH', data, 4)
    cube = bool(flags & 0x4)
    if fmt in (0x0B, 0x0C, 0x0D):
        bs = 8 if fmt == 0x0B else 16
        size = max(1, (w + 3) // 4) * max(1, (h + 3) // 4) * bs
    else:
        bpp = {0x01: 4, 0x02: 3, 0x03: 2, 0x04: 1}.get(fmt)
        if bpp is None:
            raise ValueError('IWI format 0x%x' % fmt)
        size = w * h * bpp
    faces = 6 if cube else 1
    start = len(data) - size * faces
    if start < 28:
        raise ValueError('short IWI')
    px = data[start:start + size]
    if fmt in (0x0B, 0x0C, 0x0D):
        fourcc = {0x0B: b'DXT1', 0x0C: b'DXT3', 0x0D: b'DXT5'}[fmt]
        hdr = struct.pack('<4s7I44xII4s5I5I', b'DDS ', 124, 0x1007 | 0x80000, h, w, size, 0, 1,
                          32, 4, fourcc, 0, 0, 0, 0, 0, 0x1000, 0, 0, 0, 0)
        im = Image.open(io.BytesIO(hdr + px)); im.load()
        im = im.convert('RGBA')
    elif fmt == 0x01:
        im = Image.frombuffer('RGBA', (w, h), px, 'raw', 'BGRA', 0, 1)
    elif fmt == 0x02:
        im = Image.frombuffer('RGB', (w, h), px, 'raw', 'BGR', 0, 1)
    elif fmt == 0x03:
        im = Image.frombuffer('LA', (w, h), px, 'raw', 'LA', 0, 1)
    else:
        im = Image.frombuffer('L', (w, h), px, 'raw', 'L', 0, 1)
    if im.mode == 'RGBA' and im.getextrema()[3][0] == 255:
        im = im.convert('RGB')
    return im


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('game_dir'); ap.add_argument('map'); ap.add_argument('out_dir')
    ap.add_argument('--no-textures', action='store_true')
    ap.add_argument('--all-maps', action='store_true', help='also convert normal and specular maps')
    ap.add_argument('--time-limit', type=float, default=None, help='stop converting textures after N seconds (rerun to continue)')
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    ex = Exporter(a.game_dir, a.map, a.out_dir)
    ex.run()
    pending = 0
    if not a.no_textures:
        kinds = ('colorMap', 'normalMap', 'specularMap') if a.all_maps else ('colorMap',)
        missing, pending = ex.convert_textures(kinds, a.time_limit)
        ex.link_textures()
        if missing:
            print('missing textures:', ', '.join(missing[:40]), '...' if len(missing) > 40 else '')
    ex.write()
    print('world:', ex.stats, 'bounds', ex.world_bounds)
    print('meshes %d, nodes %d, materials %d, bin %.1f MB' % (len(ex.meshes), len(ex.nodes), len(ex.materials), len(ex.bin) / 1e6))
    if ex.missing_models:
        print('models not found in %s or fallback zones:' % a.map, dict(ex.missing_models))
    if pending:
        print('PENDING textures: %d (run again to continue)' % pending)


if __name__ == '__main__':
    main()
