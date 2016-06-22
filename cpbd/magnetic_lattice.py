from ocelot.cpbd.optics import MethodTM
from ocelot.cpbd.elements import *
from ocelot.common.logging import *
from copy import deepcopy
logger = Logger()





flatten = lambda *n: (e for a in n
                      for e in (flatten(*a) if isinstance(a, (tuple, list)) else (a,)))


class MagneticLattice:
    def __init__(self, sequence, start=None, stop=None, method=MethodTM()):
        #self.energy = energy
        self.sequence = list(flatten(sequence))
        self.method = method
        try:
            if start != None:
                id1 = self.sequence.index(start)
            else:
                id1 = 0
            if stop != None:
                id2 = self.sequence.index(stop) + 1
                self.sequence = self.sequence[id1:id2]
            else:
                self.sequence = self.sequence[id1:]
        except:
            print('cannot construct sequence, element not found')
            raise

        #self.transferMaps = {}
        # create transfer map and calculate lattice length
        self.totalLen = 0
        if not self.check_edges():
            self.add_edges()
        self.update_transfer_maps()

        self.__hash__ = {}
        #print 'creating hash'
        for e in self.sequence:
            #print e
            self.__hash__[e] = e

    def __getitem__(self, el):
        try:
            return self.__hash__[el]
        except:
            return None

    def check_edges(self):
        """
        if there are edges on the ends of dipoles return True, else False
        """
        for i in range(len(self.sequence)-2):
            prob_edge1 = self.sequence[i]
            elem = self.sequence[i+1]
            prob_edge2 = self.sequence[i+2]
            if elem.__class__ in (SBend, RBend, Bend):  # , "hcor", "vcor"
                if prob_edge1.__class__ != Edge and prob_edge2.__class__ != Edge:
                    #print elem.type, prob_edge1.type, prob_edge2.type
                    return False
        return True

    def add_edges(self):
        n = 0
        for i in range(len(self.sequence)):
            elem = self.sequence[n]
            if elem.__class__ in (SBend, RBend, Bend) and elem.l != 0.:  # , "hcor", "vcor"

                e_name = elem.id

                if elem.id == None:
                    e_name = "b_" + str(i)

                e1 = Edge(l=elem.l, angle=elem.angle, k1=elem.k1, edge=elem.e1, tilt=elem.tilt, dtilt=elem.dtilt,
                          dx=elem.dx, dy=elem.dy, h_pole=elem.h_pole1, gap=elem.gap, fint=elem.fint1, pos=1,
                          eid=e_name + "_e1")

                self.sequence.insert(n, e1)

                e2 = Edge(l=elem.l, angle=elem.angle, k1=elem.k1, edge=elem.e2, tilt=elem.tilt, dtilt=elem.dtilt,
                          dx=elem.dx, dy=elem.dy, h_pole=elem.h_pole2, gap=elem.gap, fint=elem.fint2, pos=2,
                          eid=e_name + "_e2")

                self.sequence.insert(n+2, e2)
                n += 2
            n += 1

    def update_transfer_maps(self):
        #E = self.energy
        self.totalLen = 0
        for element in self.sequence:
            if element.__class__ == Undulator:
                if element.field_file != None:
                    element.l = element.field_map.l * element.field_map.field_file_rep
                    if element.field_map.units == "mm":
                        element.l = element.l*0.001
            self.totalLen += element.l
            #print(element.k1)
            element.transfer_map = self.method.create_tm(element)
            logger.debug("update: " + element.transfer_map.__class__.__name__)
            #print("update: ", element.transfer_map.__class__.__name__)
            if 'pulse' in element.__dict__: element.transfer_map.pulse = element.pulse
        return self

    def printElements(self):
        print('\nLattice\n')
        for e in self.sequence:
            print('-->',  e.id, '[', e.l, ']')