OPENQASM 2.0;
include "qelib1.inc";


// Qubits: [q(0), q(1), q(2), q(3)]
qreg q[4];


h q[0];
h q[1];
h q[2];
h q[3];
rz(pi*-0.75) q[0];
rz(pi*-0.75) q[1];
rz(pi*-0.75) q[2];
rz(pi*-0.75) q[3];
h q[0];
h q[1];
h q[2];
h q[3];
cx q[0],q[1];
rz(pi*-0.5) q[1];
cx q[0],q[1];
cx q[1],q[2];
rz(pi*-0.5) q[2];
cx q[1],q[2];
cx q[2],q[3];
rz(pi*-0.5) q[3];
cx q[2],q[3];
