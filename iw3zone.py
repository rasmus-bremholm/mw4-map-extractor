"""Generic IW3 (CoD4) fastfile zone walker.

Driven by struct layouts (iw3types.json, produced from the documented IW3 asset
structs) plus the zone command files that describe counts / conditions / blocks.
It emulates the game's sequential loader so that every struct in the stream can
be located and alias pointers can be resolved.
"""
import json, re, struct, zlib, bisect, os, sys

TEMP, RUNTIME, LARGE_RUNTIME, PHYS_RUNTIME, VIRTUAL, LARGE, PHYSICAL, VERTEX, INDEX = range(9)
BLOCKS = {'XFILE_BLOCK_TEMP': 0, 'XFILE_BLOCK_RUNTIME': 1, 'XFILE_BLOCK_LARGE_RUNTIME': 2,
          'XFILE_BLOCK_PHYSICAL_RUNTIME': 3, 'XFILE_BLOCK_VIRTUAL': 4, 'XFILE_BLOCK_LARGE': 5,
          'XFILE_BLOCK_PHYSICAL': 6, 'XFILE_BLOCK_VERTEX': 7, 'XFILE_BLOCK_INDEX': 8}
BTYPE = ['temp', 'runtime', 'runtime', 'runtime', 'normal', 'normal', 'normal', 'normal', 'normal']
FOLLOWING, INSERT = 0xFFFFFFFF, 0xFFFFFFFE

ASSET_TYPES = {1: 'PhysPreset', 2: 'XAnimParts', 3: 'XModel', 4: 'Material', 5: 'MaterialTechniqueSet',
               6: 'GfxImage', 7: 'snd_alias_list_t', 8: 'SndCurve', 9: 'LoadedSound', 10: 'clipMap_t',
               11: 'clipMap_t', 12: 'ComWorld', 13: 'GameWorldSp', 14: 'GameWorldMp', 15: 'MapEnts',
               16: 'GfxWorld', 17: 'GfxLightDef', 19: 'Font_s', 20: 'MenuList', 21: 'menuDef_t',
               22: 'LocalizeEntry', 23: 'WeaponDef', 25: 'FxEffectDef', 26: 'FxImpactTable',
               31: 'RawFile', 32: 'StringTable'}
ASSET_STRUCTS = set(ASSET_TYPES.values())

PRIMFMT = {'char': 'b', 'signed char': 'b', 'unsigned char': 'B', 'bool': 'B', 'short': 'h',
           'unsigned short': 'H', 'int': 'i', 'unsigned int': 'I', 'long': 'i', 'unsigned long': 'I',
           'float': 'f', 'long long': 'q', 'unsigned long long': 'Q'}


class MProps:
    __slots__ = ('string', 'count', 'arraysize', 'reusable', 'cond', 'block', 'allocalign')

    def __init__(self):
        self.string = False; self.count = {}; self.arraysize = None; self.reusable = False
        self.cond = None; self.block = None; self.allocalign = None


class SProps:
    def __init__(self):
        self.block = None; self.allocalign = None; self.order = None; self.m = {}


class Spec:
    def __init__(self, types_json, cmd_dir):
        j = json.load(open(types_json))
        self.structs = j['structs']; self.enums = j['enums']
        for n, s in self.structs.items():
            s['byname'] = {f['name']: f for f in s['fields'] if f['name']}
        self.props = {n: SProps() for n in self.structs}
        self._leaf = {}; self._dyn = {}
        for fn in sorted(os.listdir(cmd_dir)):
            if fn.endswith('.txt'):
                self.parse_cmds(open(os.path.join(cmd_dir, fn)).read())

    # ---- field lookup (transparent through unnamed anonymous members)
    def find(self, sn, name):
        s = self.structs[sn]
        f = s['byname'].get(name)
        if f is not None:
            return sn, f, 0
        for g in s['fields']:
            if not g['name'] and g['base']['k'] == 'struct' and not g['mods']:
                r = self.find(g['base']['n'], name)
                if r:
                    return r[0], r[1], r[2] + g['off']
        return None

    def mp(self, sn, fname):
        p = self.props[sn].m.get(fname)
        if p is None:
            p = self.props[sn].m[fname] = MProps()
        return p

    # ---- command files
    def resolve_target(self, use, path):
        """returns (ctxStruct, ownerStruct, field, indices)"""
        comps = path.split('::')
        idx = ()
        m = re.match(r'^(\w+)((?:\[\d+\])*)$', comps[-1])
        comps[-1] = m.group(1)
        if m.group(2):
            idx = tuple(int(x) for x in re.findall(r'\[(\d+)\]', m.group(2)))
        ctx = use
        if not (use and self.find(use, comps[0])) and comps[0] in self.structs:
            ctx = comps[0]; comps = comps[1:]
            if not comps:
                return ctx, ctx, None, idx
        cur = ctx
        for c in comps[:-1]:
            o, f, _ = self.find(cur, c)
            cur = f['base']['n']
        r = self.find(cur, comps[-1])
        if r is None:
            raise KeyError('%s in %s' % (path, ctx))
        return ctx, r[0], r[1], idx

    def parse_cmds(self, text):
        text = re.sub(r'//[^\n]*', '', text)
        use = None
        for st in text.split(';'):
            st = ' '.join(st.split())
            if not st:
                continue
            if st.startswith('use '):
                use = st[4:].strip(); continue
            if st.startswith('reorder'):
                head, body = st.split(':', 1)
                tn = head[7:].strip() or use
                self.props[tn].order = body.split()
                continue
            if not st.startswith('set '):
                continue
            w = st.split(' ', 3)
            kind = w[1]
            if kind in ('action', 'assetref', 'name'):
                continue
            if kind == 'block':
                if len(w) == 3:
                    self.props[use].block = BLOCKS[w[2]]
                else:
                    ctx, own, f, idx = self.resolve_target(use, w[2])
                    self.mp(own, f['name']).block = BLOCKS[w[3]]
                continue
            ctx, own, f, idx = self.resolve_target(use, w[2])
            if f is None:
                if kind == 'allocalign':
                    self.props[own].allocalign = int(w[3])
                continue
            p = self.mp(own, f['name'])
            if kind in ('string',):
                p.string = True
            elif kind == 'scriptstring':
                pass
            elif kind == 'reusable':
                p.reusable = True
            elif kind == 'count':
                p.count[idx] = self.expr(ctx, w[3])
            elif kind == 'arraysize':
                p.arraysize = self.expr(ctx, w[3])
            elif kind == 'condition':
                if w[3] == 'never':
                    p.cond = False
                elif w[3] == 'always':
                    p.cond = None
                else:
                    p.cond = self.expr(ctx, w[3])
            elif kind == 'allocalign':
                p.allocalign = int(w[3])
            else:
                raise ValueError(st)

    # ---- expressions -> python closures f(zone)
    def expr(self, ctx, s):
        toks = re.findall(r'0x[0-9a-fA-F]+|\d+|[A-Za-z_]\w*(?:::\w+)*|\|\||&&|==|!=|<=|>=|<<|>>|[-+*/%()<>\[\]!]', s)
        pos = [0]

        def peek():
            return toks[pos[0]] if pos[0] < len(toks) else None

        def nxt():
            t = toks[pos[0]]; pos[0] += 1; return t

        def prim():
            t = nxt()
            if t == '(':
                e = binop(0); nxt(); return e
            if t == '-':
                e = prim(); return lambda z: -e(z)
            if t == '!':
                e = prim(); return lambda z: int(not e(z))
            if t[0].isdigit():
                v = int(t, 0); return lambda z: v
            if t in self.enums:
                v = self.enums[t]; return lambda z: v
            comps = t.split('::')
            base = ctx
            if not self.find(ctx, comps[0]):
                base = comps[0]; comps = comps[1:]
            steps = []  # (offset, idxexpr, elemsize)
            cur = base; off = 0; f = None
            for i, c in enumerate(comps):
                o, f, extra = self.find(cur, c)
                off += extra + f['off']
                if f['base']['k'] == 'struct':
                    cur = f['base']['n']
            idxs = []
            while peek() == '[':
                nxt(); idxs.append(binop(0)); nxt()
            b = f['base']
            dims = [m[1] for m in f['mods'] if m[0] == 'arr']
            if any(m[0] == 'ptr' for m in f['mods']):
                fmt = 'I'; esz = 4
            else:
                fmt = PRIMFMT[b['c']]; esz = b['size']
            bits = f.get('bits'); bitoff = f.get('bitoff', 0) % 8 if bits else 0
            strides = []
            for k in range(len(idxs)):
                st = esz
                for d in dims[k + 1:]:
                    st *= d
                strides.append(st)
            fmt = '<' + fmt

            def ev(z):
                a = z.cur[base] + off
                for ie, st in zip(idxs, strides):
                    a += ie(z) * st
                v = struct.unpack_from(fmt, z.d, a)[0]
                if bits:
                    v = (v >> bitoff) & ((1 << bits) - 1)
                return v
            return ev

        PREC = [('||',), ('&&',), ('==', '!='), ('<', '>', '<=', '>='), ('<<', '>>'), ('+', '-'), ('*', '/', '%')]
        OPS = {'||': lambda a, b: int(bool(a) or bool(b)), '&&': lambda a, b: int(bool(a) and bool(b)),
               '==': lambda a, b: int(a == b), '!=': lambda a, b: int(a != b), '<': lambda a, b: int(a < b),
               '>': lambda a, b: int(a > b), '<=': lambda a, b: int(a <= b), '>=': lambda a, b: int(a >= b),
               '<<': lambda a, b: a << b, '>>': lambda a, b: a >> b, '+': lambda a, b: a + b,
               '-': lambda a, b: a - b, '*': lambda a, b: a * b, '/': lambda a, b: a // b,
               '%': lambda a, b: a % b}

        def binop(level):
            if level == len(PREC):
                return prim()
            l = binop(level + 1)
            while peek() in PREC[level]:
                op = OPS[nxt()]; r = binop(level + 1)
                l = (lambda l, r, op: lambda z: op(l(z), r(z)))(l, r, op)
            return l
        e = binop(0)
        assert pos[0] == len(toks), (s, toks[pos[0]:])
        return e

    # ---- derived properties
    def has_dyn_size(self, sn, f):
        p = self.props[sn].m.get(f['name'])
        return p is not None and p.arraysize is not None

    def is_dyn_member(self, sn, f):
        if self.has_dyn_size(sn, f):
            return True
        if any(m[0] == 'ptr' for m in f['mods']):
            return False
        return f['base']['k'] == 'struct' and self.dyn_member(f['base']['n']) is not None

    def dyn_member(self, sn):
        if sn not in self._dyn:
            self._dyn[sn] = None
            for f in self.structs[sn]['fields']:
                if self.is_dyn_member(sn, f):
                    self._dyn[sn] = f; break
        return self._dyn[sn]

    def ignored(self, sn, f):
        p = self.props[sn].m.get(f['name'])
        return p is not None and p.cond is False

    def is_leaf(self, sn):
        if sn in self._leaf:
            return self._leaf[sn]
        self._leaf[sn] = True
        r = True
        for f in self.structs[sn]['fields']:
            if self.ignored(sn, f):
                continue
            if any(m[0] == 'ptr' for m in f['mods']) or self.has_dyn_size(sn, f):
                r = False; break
            if f['base']['k'] == 'struct' and not self.is_leaf(f['base']['n']):
                r = False; break
        if self.props[sn].block is not None:
            r = False
        self._leaf[sn] = r
        return r

    def ordered(self, sn):
        s = self.structs[sn]
        o = self.props[sn].order
        if not o:
            return s['fields']
        listed = [x for x in o if x != '...']
        rest = [f for f in s['fields'] if f['name'] not in listed]
        lf = [s['byname'][x] for x in listed]
        if '...' in o:
            i = o.index('...')
            names = [f['name'] for f in s['fields']]
            piv = names.index(o[i + 1])
            pre = [f for k, f in enumerate(s['fields']) if k < piv and f['name'] not in listed]
            post = [f for k, f in enumerate(s['fields']) if k > piv and f['name'] not in listed]
            return lf[:i] + pre + lf[i:] + post
        return lf + rest


class Zone:
    def __init__(self, spec, zone_bytes, verbose=False):
        self.sp = spec; self.d = zone_bytes; self.verbose = verbose
        self.pos = 0
        self.boff = [0] * 9; self.bstack = []; self.tstack = []
        self.segs = {b: ([], []) for b in range(9)}
        self.ptr = {}       # stream addr of pointer field -> result
        self.inserted = {}  # VIRTUAL block offset of inserted pointer -> result
        self.cur = {}
        self.assets = []    # (struct name, stream offset, type id)
        self.depth = 0

    # ---- stream primitives
    def u32(self, a):
        return struct.unpack_from('<I', self.d, a)[0]

    def push(self, b):
        self.bstack.append(b)
        if b == TEMP:
            self.tstack.append(self.boff[TEMP])

    def pop(self):
        b = self.bstack.pop()
        if b == TEMP:
            self.boff[TEMP] = self.tstack.pop()

    def alloc(self, align):
        b = self.bstack[-1]
        if align > 1:
            self.boff[b] = (self.boff[b] + align - 1) // align * align

    def load(self, size):
        b = self.bstack[-1]
        if BTYPE[b] == 'runtime':
            self.boff[b] += size
            return None
        a = self.pos
        self.pos += size
        if self.pos > len(self.d):
            raise EOFError('read past end of zone at %d' % a)
        if BTYPE[b] == 'normal':
            s = self.segs[b]; s[0].append(self.boff[b]); s[1].append((size, a))
        self.boff[b] += size
        return a

    def load_cstr(self):
        b = self.bstack[-1]
        a = self.pos
        e = self.d.index(b'\0', a)
        size = e + 1 - a
        self.pos = e + 1
        if BTYPE[b] == 'normal':
            s = self.segs[b]; s[0].append(self.boff[b]); s[1].append((size, a))
        self.boff[b] += size
        return a

    def insert_ptr(self):
        self.boff[VIRTUAL] = (self.boff[VIRTUAL] + 3) // 4 * 4
        o = self.boff[VIRTUAL]
        self.boff[VIRTUAL] += 4
        return o

    def resolve(self, p):
        """zone pointer -> stream offset (or None)"""
        v = p - 1
        b = v >> 28; o = v & 0x0FFFFFFF
        offs, meta = self.segs[b]
        i = bisect.bisect_right(offs, o) - 1
        if i < 0:
            return None
        size, a = meta[i]
        if o - offs[i] >= size:
            return None
        return a + (o - offs[i])

    def alias(self, p):
        """zone pointer to a pointer slot -> what that slot points at"""
        v = p - 1
        r = None
        if (v >> 28) == VIRTUAL:
            r = self.inserted.get(v & 0x0FFFFFFF)
        if r is None:
            sa = self.resolve(p)
            if sa is not None:
                r = self.ptr.get(sa)
        return r if r is not None else ('badalias', p)

    def cstr(self, a):
        e = self.d.index(b'\0', a)
        return self.d[a:e].decode('latin1')

    # ---- loaders
    def xstring(self, addr):
        p = self.u32(addr)
        if p == 0:
            return None
        if p == FOLLOWING:
            self.alloc(1)
            a = self.load_cstr()
        else:
            a = self.resolve(p)
        r = ('str', a)
        self.ptr[addr] = r
        return r

    def asset_ptr(self, addr, tn, tid=None):
        P = self.sp.props[tn]
        intemp = P.block == TEMP
        if intemp:
            self.push(TEMP)
        p = self.u32(addr)
        r = None
        if p:
            if p == FOLLOWING or (intemp and p == INSERT):
                self.alloc(P.allocalign or self.sp.structs[tn]['align'])
                slot = self.insert_ptr() if (intemp and p == INSERT) else None
                a = self.struct(tn, None, True)
                r = ('asset', tn, a)
                self.assets.append((tn, a, tid))
                if slot is not None:
                    self.inserted[slot] = r
            elif intemp:
                r = self.alias(p)
            else:
                r = ('asset', tn, self.resolve(p))
            self.ptr[addr] = r
        if intemp:
            self.pop()
        return r

    def struct(self, tn, a, at_start):
        sp = self.sp
        S = sp.structs[tn]; P = sp.props[tn]
        dyn = sp.dyn_member(tn)
        isunion = S['kind'] == 'union'
        if at_start:
            if isunion and dyn is not None:
                a = self.pos
            elif dyn is not None:
                a = self.load(dyn['off'])
            else:
                a = self.load(S['size'])
        if a is None:
            return None
        self.cur[tn] = a
        pushed = False
        if tn in ASSET_STRUCTS:
            self.push(VIRTUAL); pushed = True
        elif P.block is not None:
            self.push(P.block); pushed = True
        fields = sp.ordered(tn)
        if isunion:
            used = []
            for f in fields:
                if sp.ignored(tn, f):
                    continue
                if dyn is not None or self.treat(tn, f):
                    used.append(f)
            chosen = None
            for f in used:
                p = P.m.get(f['name'])
                if p is not None and p.cond is not None and p.cond(self):
                    chosen = f; break
            if chosen is None:
                for f in used:
                    p = P.m.get(f['name'])
                    if p is None or p.cond is None:
                        chosen = f; break
            if chosen is not None:
                self.member(tn, a, chosen, True)
        else:
            for f in fields:
                if sp.ignored(tn, f) or not self.treat(tn, f):
                    continue
                p = P.m.get(f['name'])
                if p is not None and p.cond is not None and not p.cond(self):
                    continue
                self.member(tn, a, f, False)
        if pushed:
            self.pop()
        return a

    def treat(self, tn, f):
        sp = self.sp
        if any(m[0] == 'ptr' for m in f['mods']):
            return True
        if f['base']['k'] == 'struct' and not sp.is_leaf(f['base']['n']):
            return True
        return sp.is_dyn_member(tn, f)

    def member(self, tn, a, f, parent_dyn_union):
        sp = self.sp
        P = sp.props[tn].m.get(f['name']) or _DEFAULT
        after_partial = sp.is_dyn_member(tn, f) or (parent_dyn_union and sp.dyn_member(tn) is not None)
        blk = P.block
        push = blk is not None and blk != VIRTUAL
        if push:
            self.push(blk)
        self.ref(tn, f, P, 0, (), a + f['off'], after_partial, blk)
        if push:
            self.pop()

    def elem_size(self, f, i):
        """size/align of the type described by mods[i:] over base"""
        mods = f['mods'][i:]
        if mods and mods[0][0] == 'ptr':
            return 4, 4
        size = f['base']['size']; align = f['base']['align']
        for m in mods:
            if m[0] == 'ptr':
                size = 4; align = 4; break
        n = 1
        for m in mods:
            if m[0] == 'ptr':
                break
            n *= m[1]
        return size * n, align

    def count_for(self, P, idxs):
        c = P.count.get(idxs)
        if c is None:
            c = P.count.get(())
        return c

    def ref(self, tn, f, P, i, idxs, addr, after_partial, blk):
        sp = self.sp
        mods = f['mods']
        base = f['base']
        isstruct = base['k'] == 'struct'
        bn = base['n']
        if i >= len(mods):
            # embedded
            if isstruct and not sp.is_leaf(bn):
                self.struct(bn, addr, after_partial)
            elif after_partial:
                self.load(base['size'])
            return
        m = mods[i]
        rest = mods[i + 1:]
        if m[0] == 'arr' and i == 0 and P.arraysize is not None:
            n = P.arraysize(self)
            esz, _ = self.elem_size(f, 1)
            if isstruct and not rest and not sp.is_leaf(bn):
                a = self.load(esz * n)
                for k in range(n):
                    self.struct(bn, a + k * esz, False)
            else:
                self.load(esz * n)
            return
        if m[0] == 'ptr':
            cexp = self.count_for(P, idxs)
            if not any(x[0] == 'ptr' for x in rest):
                # single / array pointer
                if P.string:
                    self.xstring(addr); return
                if isstruct and bn in ASSET_STRUCTS and not rest:
                    self.asset_ptr(addr, bn); return
                p = self.u32(addr)
                if p == 0:
                    return
                intemp = blk == TEMP
                if P.reusable and not (p == FOLLOWING or (intemp and p == INSERT)):
                    if intemp:
                        self.ptr[addr] = self.alias(p)
                    else:
                        self.ptr[addr] = ('alias', bn, self.resolve(p))
                    return
                n = cexp(self) if cexp is not None else 1
                esz, ealign = self.elem_size(f, i + 1)
                self.alloc(P.allocalign or ealign)
                slot = self.insert_ptr() if (intemp and p == INSERT) else None
                if isstruct and not rest and sp.dyn_member(bn) is not None:
                    a = self.struct(bn, None, True)
                    r = ('data', bn, a, 1, esz)
                    self.ptr[addr] = r
                    if slot is not None:
                        self.inserted[slot] = r
                    return
                a = self.load(esz * n)
                r = ('data', bn, a, n, esz)
                self.ptr[addr] = r
                if slot is not None:
                    self.inserted[slot] = r
                if a is not None and isstruct and not rest and not sp.is_leaf(bn) and blk not in (RUNTIME, LARGE_RUNTIME, PHYS_RUNTIME):
                    for k in range(n):
                        self.struct(bn, a + k * esz, False)
                return
            # T** : pointer to array of pointers
            p = self.u32(addr)
            if p == 0:
                return
            if P.reusable and p != FOLLOWING:
                self.ptr[addr] = ('alias', 'ptr', self.resolve(p)); return
            n = cexp(self) if cexp is not None else 1
            self.alloc(P.allocalign or 4)
            a = self.load(4 * n)
            self.ptr[addr] = ('ptrs', bn, a, n)
            if a is not None:
                for k in range(n):
                    self.ptr_elem(f, P, a + 4 * k)
            return
        # array
        n = m[1]
        if not rest:
            esz = base['size']
            if isstruct and not sp.is_leaf(bn):
                if after_partial:
                    a = self.load(esz * n)
                    addr = a
                for k in range(n):
                    self.struct(bn, addr + k * esz, False)
            elif after_partial:
                self.load(esz * n)
            return
        if rest[0][0] == 'ptr' and len(rest) == 1 and not P.count:
            for k in range(n):
                self.ptr_elem(f, P, addr + 4 * k)
            return
        esz, _ = self.elem_size(f, i + 1)
        for k in range(n):
            self.ref(tn, f, P, i + 1, idxs + (k,), addr + k * esz, after_partial, blk)

    def ptr_elem(self, f, P, addr):
        sp = self.sp
        base = f['base']; bn = base['n']
        if P.string:
            self.xstring(addr); return
        p = self.u32(addr)
        if p == 0:
            return
        if base['k'] == 'struct' and bn in ASSET_STRUCTS:
            self.asset_ptr(addr, bn); return
        if P.reusable and p != FOLLOWING:
            self.ptr[addr] = ('alias', bn, self.resolve(p)); return
        self.alloc(base['align'])
        if base['k'] == 'struct' and not sp.is_leaf(bn):
            a = self.struct(bn, None, True)
        else:
            a = self.load(base['size'])
        self.ptr[addr] = ('data', bn, a, 1, base['size'])

    def run(self):
        d = self.d
        hdr = struct.unpack_from('<11I', d, 0)
        self.block_sizes = hdr[2:]
        self.pos = 44
        nstr, pstr, nasset, passet = struct.unpack_from('<IIII', d, self.pos)
        self.pos += 16
        self.push(VIRTUAL)
        self.script_strings = []
        if pstr:
            self.alloc(4)
            a = self.load(4 * nstr)
            for k in range(nstr):
                r = self.xstring(a + 4 * k)
                self.script_strings.append(self.cstr(r[1]) if r and r[1] is not None else None)
        self.alloc(4)
        a = self.load(8 * nasset)
        self.asset_list = a; self.nasset = nasset
        for k in range(nasset):
            t = self.u32(a + 8 * k)
            tn = ASSET_TYPES[t]
            try:
                self.asset_ptr(a + 8 * k + 4, tn, t)
            except Exception:
                sys.stderr.write('FAILED in asset %d/%d type %s at stream %d\n' % (k, nasset, tn, self.pos))
                raise
            if self.verbose:
                print(k, tn, self.pos, self.boff)
        self.pop()

    # ---- reading helpers for extraction
    def field(self, tn, a, path):
        """returns (stream addr, field dict) for a '::' separated member path"""
        cur = tn
        f = None
        for c in path.split('::'):
            o, f, extra = self.sp.find(cur, c)
            a += extra + f['off']
            if f['base']['k'] == 'struct':
                cur = f['base']['n']
        return a, f

    def get(self, tn, a, path, fmt=None):
        addr, f = self.field(tn, a, path)
        if fmt is None:
            if any(m[0] == 'ptr' for m in f['mods']):
                fmt = 'I'
            else:
                fmt = PRIMFMT[f['base']['c']]
        v = struct.unpack_from('<' + fmt, self.d, addr)
        return v[0] if len(v) == 1 else v

    def follow(self, tn, a, path):
        addr, f = self.field(tn, a, path)
        return self.ptr.get(addr)

    def string(self, tn, a, path):
        r = self.follow(tn, a, path)
        if r and r[0] == 'str' and r[1] is not None:
            return self.cstr(r[1])
        return None


_DEFAULT = MProps()


def load_ff(path):
    d = open(path, 'rb').read()
    assert d[:8] == b'IWffu100', d[:8]
    return zlib.decompress(d[12:])
