��
�"�"
D
AddV2
x"T
y"T
z"T"
Ttype:
2	��
h
All	
input

reduction_indices"Tidx

output
"
	keep_dimsbool( "
Tidxtype0:
2	
h
Any	
input

reduction_indices"Tidx

output
"
	keep_dimsbool( "
Tidxtype0:
2	
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( �
p
BatchMatMulV2
x"T
y"T
output"T"
Ttype:
2	"
adj_xbool( "
adj_ybool( 
�
BiasAdd

value"T	
bias"T
output"T""
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
$
DisableCopyOnRead
resource�
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
.
Identity

input"T
output"T"	
Ttype
u
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
2	
�
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( �
?
Mul
x"T
y"T
z"T"
Ttype:
2	�

NoOp
U
NotEqual
x"T
y"T
z
"	
Ttype"$
incompatible_shape_errorbool(�
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype�
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
d
Shape

input"T&
output"out_type��out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
9
Softmax
logits"T
softmax"T"
Ttype:
2
[
Split
	split_dim

value"T
output"T*	num_split"
	num_splitint(0"	
Ttype
�
SplitV

value"T
size_splits"Tlen
	split_dim
output"T*	num_split"
	num_splitint(0"	
Ttype"
Tlentype0	:
2	
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ��
@
StaticRegexFullMatch	
input

output
"
patternstring
�
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
L

StringJoin
inputs*N

output"

Nint("
	separatorstring 
<
Sub
x"T
y"T
z"T"
Ttype:
2	
-
Tanh
x"T
y"T"
Ttype:

2
�
TensorListFromTensor
tensor"element_dtype
element_shape"
shape_type/
output_handle���element_dtype"
element_dtypetype"

shape_typetype:
2	
�
TensorListReserve
element_shape"
shape_type
num_elements(
handle���element_dtype"
element_dtypetype"

shape_typetype:
2	
�
TensorListStack
input_handle
element_shape
tensor"element_dtype"
element_dtypetype" 
num_elementsint���������
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
P
Unpack

value"T
output"T*num"
numint("	
Ttype"
axisint 
�
VarHandleOp
resource"
	containerstring "
shared_namestring "

debug_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �
�
While

input2T
output2T"
T
list(type)("
condfunc"
bodyfunc" 
output_shapeslist(shape)
 "
parallel_iterationsint
�
&
	ZerosLike
x"T
y"T"	
Ttype"serve*2.15.02v2.15.0-rc1-8-g6887368d6d48��
�
!training/Adam/gru/gru_cell/bias/vVarHandleOp*
_output_shapes
: *2

debug_name$"training/Adam/gru/gru_cell/bias/v/*
dtype0*
shape
:`*2
shared_name#!training/Adam/gru/gru_cell/bias/v
�
5training/Adam/gru/gru_cell/bias/v/Read/ReadVariableOpReadVariableOp!training/Adam/gru/gru_cell/bias/v*
_output_shapes

:`*
dtype0
�
-training/Adam/gru/gru_cell/recurrent_kernel/vVarHandleOp*
_output_shapes
: *>

debug_name0.training/Adam/gru/gru_cell/recurrent_kernel/v/*
dtype0*
shape
: `*>
shared_name/-training/Adam/gru/gru_cell/recurrent_kernel/v
�
Atraining/Adam/gru/gru_cell/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp-training/Adam/gru/gru_cell/recurrent_kernel/v*
_output_shapes

: `*
dtype0
�
#training/Adam/gru/gru_cell/kernel/vVarHandleOp*
_output_shapes
: *4

debug_name&$training/Adam/gru/gru_cell/kernel/v/*
dtype0*
shape
:`*4
shared_name%#training/Adam/gru/gru_cell/kernel/v
�
7training/Adam/gru/gru_cell/kernel/v/Read/ReadVariableOpReadVariableOp#training/Adam/gru/gru_cell/kernel/v*
_output_shapes

:`*
dtype0
�
training/Adam/dense/bias/vVarHandleOp*
_output_shapes
: *+

debug_nametraining/Adam/dense/bias/v/*
dtype0*
shape:*+
shared_nametraining/Adam/dense/bias/v
�
.training/Adam/dense/bias/v/Read/ReadVariableOpReadVariableOptraining/Adam/dense/bias/v*
_output_shapes
:*
dtype0
�
training/Adam/dense/kernel/vVarHandleOp*
_output_shapes
: *-

debug_nametraining/Adam/dense/kernel/v/*
dtype0*
shape
: *-
shared_nametraining/Adam/dense/kernel/v
�
0training/Adam/dense/kernel/v/Read/ReadVariableOpReadVariableOptraining/Adam/dense/kernel/v*
_output_shapes

: *
dtype0
�
!training/Adam/gru/gru_cell/bias/mVarHandleOp*
_output_shapes
: *2

debug_name$"training/Adam/gru/gru_cell/bias/m/*
dtype0*
shape
:`*2
shared_name#!training/Adam/gru/gru_cell/bias/m
�
5training/Adam/gru/gru_cell/bias/m/Read/ReadVariableOpReadVariableOp!training/Adam/gru/gru_cell/bias/m*
_output_shapes

:`*
dtype0
�
-training/Adam/gru/gru_cell/recurrent_kernel/mVarHandleOp*
_output_shapes
: *>

debug_name0.training/Adam/gru/gru_cell/recurrent_kernel/m/*
dtype0*
shape
: `*>
shared_name/-training/Adam/gru/gru_cell/recurrent_kernel/m
�
Atraining/Adam/gru/gru_cell/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp-training/Adam/gru/gru_cell/recurrent_kernel/m*
_output_shapes

: `*
dtype0
�
#training/Adam/gru/gru_cell/kernel/mVarHandleOp*
_output_shapes
: *4

debug_name&$training/Adam/gru/gru_cell/kernel/m/*
dtype0*
shape
:`*4
shared_name%#training/Adam/gru/gru_cell/kernel/m
�
7training/Adam/gru/gru_cell/kernel/m/Read/ReadVariableOpReadVariableOp#training/Adam/gru/gru_cell/kernel/m*
_output_shapes

:`*
dtype0
�
training/Adam/dense/bias/mVarHandleOp*
_output_shapes
: *+

debug_nametraining/Adam/dense/bias/m/*
dtype0*
shape:*+
shared_nametraining/Adam/dense/bias/m
�
.training/Adam/dense/bias/m/Read/ReadVariableOpReadVariableOptraining/Adam/dense/bias/m*
_output_shapes
:*
dtype0
�
training/Adam/dense/kernel/mVarHandleOp*
_output_shapes
: *-

debug_nametraining/Adam/dense/kernel/m/*
dtype0*
shape
: *-
shared_nametraining/Adam/dense/kernel/m
�
0training/Adam/dense/kernel/m/Read/ReadVariableOpReadVariableOptraining/Adam/dense/kernel/m*
_output_shapes

: *
dtype0
�
false_negativesVarHandleOp*
_output_shapes
: * 

debug_namefalse_negatives/*
dtype0*
shape:�* 
shared_namefalse_negatives
p
#false_negatives/Read/ReadVariableOpReadVariableOpfalse_negatives*
_output_shapes	
:�*
dtype0
�
false_positivesVarHandleOp*
_output_shapes
: * 

debug_namefalse_positives/*
dtype0*
shape:�* 
shared_namefalse_positives
p
#false_positives/Read/ReadVariableOpReadVariableOpfalse_positives*
_output_shapes	
:�*
dtype0
�
true_negativesVarHandleOp*
_output_shapes
: *

debug_nametrue_negatives/*
dtype0*
shape:�*
shared_nametrue_negatives
n
"true_negatives/Read/ReadVariableOpReadVariableOptrue_negatives*
_output_shapes	
:�*
dtype0
�
true_positivesVarHandleOp*
_output_shapes
: *

debug_nametrue_positives/*
dtype0*
shape:�*
shared_nametrue_positives
n
"true_positives/Read/ReadVariableOpReadVariableOptrue_positives*
_output_shapes	
:�*
dtype0
v
countVarHandleOp*
_output_shapes
: *

debug_namecount/*
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
v
totalVarHandleOp*
_output_shapes
: *

debug_nametotal/*
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
�
training/Adam/learning_rateVarHandleOp*
_output_shapes
: *,

debug_nametraining/Adam/learning_rate/*
dtype0*
shape: *,
shared_nametraining/Adam/learning_rate
�
/training/Adam/learning_rate/Read/ReadVariableOpReadVariableOptraining/Adam/learning_rate*
_output_shapes
: *
dtype0
�
training/Adam/decayVarHandleOp*
_output_shapes
: *$

debug_nametraining/Adam/decay/*
dtype0*
shape: *$
shared_nametraining/Adam/decay
s
'training/Adam/decay/Read/ReadVariableOpReadVariableOptraining/Adam/decay*
_output_shapes
: *
dtype0
�
training/Adam/beta_2VarHandleOp*
_output_shapes
: *%

debug_nametraining/Adam/beta_2/*
dtype0*
shape: *%
shared_nametraining/Adam/beta_2
u
(training/Adam/beta_2/Read/ReadVariableOpReadVariableOptraining/Adam/beta_2*
_output_shapes
: *
dtype0
�
training/Adam/beta_1VarHandleOp*
_output_shapes
: *%

debug_nametraining/Adam/beta_1/*
dtype0*
shape: *%
shared_nametraining/Adam/beta_1
u
(training/Adam/beta_1/Read/ReadVariableOpReadVariableOptraining/Adam/beta_1*
_output_shapes
: *
dtype0
�
training/Adam/iterVarHandleOp*
_output_shapes
: *#

debug_nametraining/Adam/iter/*
dtype0	*
shape: *#
shared_nametraining/Adam/iter
q
&training/Adam/iter/Read/ReadVariableOpReadVariableOptraining/Adam/iter*
_output_shapes
: *
dtype0	
�
gru/gru_cell/biasVarHandleOp*
_output_shapes
: *"

debug_namegru/gru_cell/bias/*
dtype0*
shape
:`*"
shared_namegru/gru_cell/bias
w
%gru/gru_cell/bias/Read/ReadVariableOpReadVariableOpgru/gru_cell/bias*
_output_shapes

:`*
dtype0
�
gru/gru_cell/recurrent_kernelVarHandleOp*
_output_shapes
: *.

debug_name gru/gru_cell/recurrent_kernel/*
dtype0*
shape
: `*.
shared_namegru/gru_cell/recurrent_kernel
�
1gru/gru_cell/recurrent_kernel/Read/ReadVariableOpReadVariableOpgru/gru_cell/recurrent_kernel*
_output_shapes

: `*
dtype0
�
gru/gru_cell/kernelVarHandleOp*
_output_shapes
: *$

debug_namegru/gru_cell/kernel/*
dtype0*
shape
:`*$
shared_namegru/gru_cell/kernel
{
'gru/gru_cell/kernel/Read/ReadVariableOpReadVariableOpgru/gru_cell/kernel*
_output_shapes

:`*
dtype0
�

dense/biasVarHandleOp*
_output_shapes
: *

debug_namedense/bias/*
dtype0*
shape:*
shared_name
dense/bias
e
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes
:*
dtype0
�
dense/kernelVarHandleOp*
_output_shapes
: *

debug_namedense/kernel/*
dtype0*
shape
: *
shared_namedense/kernel
m
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes

: *
dtype0
�
serving_default_input_priorPlaceholder*,
_output_shapes
:����������
*
dtype0*!
shape:����������

�
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_priorgru/gru_cell/biasgru/gru_cell/kernelgru/gru_cell/recurrent_kerneldense/kernel
dense/bias*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8� *+
f&R$
"__inference_signature_wrapper_2544

NoOpNoOp
�I
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�H
value�HB�H B�H
�
layer-0
layer-1
layer-2
layer-3
layer-4
layer-5
layer-6
layer_with_weights-0
layer-7
	layer_with_weights-1
	layer-8

	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures*
* 
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses* 
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
axes* 
�
 	variables
!trainable_variables
"regularization_losses
#	keras_api
$__call__
*%&call_and_return_all_conditional_losses* 
�
&	variables
'trainable_variables
(regularization_losses
)	keras_api
*__call__
*+&call_and_return_all_conditional_losses
,axes* 
�
-	variables
.trainable_variables
/regularization_losses
0	keras_api
1__call__
*2&call_and_return_all_conditional_losses* 
�
3	variables
4trainable_variables
5regularization_losses
6	keras_api
7__call__
*8&call_and_return_all_conditional_losses* 
�
9	variables
:trainable_variables
;regularization_losses
<	keras_api
=__call__
*>&call_and_return_all_conditional_losses
?_random_generator
@cell
A
state_spec*
�
B	variables
Ctrainable_variables
Dregularization_losses
E	keras_api
F__call__
*G&call_and_return_all_conditional_losses

Hkernel
Ibias*
'
J0
K1
L2
H3
I4*
'
J0
K1
L2
H3
I4*
* 
�
Mnon_trainable_variables

Nlayers
Ometrics
Player_regularization_losses
Qlayer_metrics

	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

Rtrace_0
Strace_1* 

Ttrace_0
Utrace_1* 
* 
�
Viter

Wbeta_1

Xbeta_2
	Ydecay
Zlearning_rateHm�Im�Jm�Km�Lm�Hv�Iv�Jv�Kv�Lv�*

[serving_default* 
* 
* 
* 
�
\non_trainable_variables

]layers
^metrics
_layer_regularization_losses
`layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses* 

atrace_0* 

btrace_0* 
* 
* 
* 
�
cnon_trainable_variables

dlayers
emetrics
flayer_regularization_losses
glayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses* 

htrace_0* 

itrace_0* 
* 
* 
* 
* 
�
jnon_trainable_variables

klayers
lmetrics
mlayer_regularization_losses
nlayer_metrics
 	variables
!trainable_variables
"regularization_losses
$__call__
*%&call_and_return_all_conditional_losses
&%"call_and_return_conditional_losses* 

otrace_0* 

ptrace_0* 
* 
* 
* 
�
qnon_trainable_variables

rlayers
smetrics
tlayer_regularization_losses
ulayer_metrics
&	variables
'trainable_variables
(regularization_losses
*__call__
*+&call_and_return_all_conditional_losses
&+"call_and_return_conditional_losses* 

vtrace_0* 

wtrace_0* 
* 
* 
* 
* 
�
xnon_trainable_variables

ylayers
zmetrics
{layer_regularization_losses
|layer_metrics
-	variables
.trainable_variables
/regularization_losses
1__call__
*2&call_and_return_all_conditional_losses
&2"call_and_return_conditional_losses* 

}trace_0* 

~trace_0* 
* 
* 
* 
�
non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
3	variables
4trainable_variables
5regularization_losses
7__call__
*8&call_and_return_all_conditional_losses
&8"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 

J0
K1
L2*

J0
K1
L2*
* 
�
�states
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
9	variables
:trainable_variables
;regularization_losses
=__call__
*>&call_and_return_all_conditional_losses
&>"call_and_return_conditional_losses*
:
�trace_0
�trace_1
�trace_2
�trace_3* 
:
�trace_0
�trace_1
�trace_2
�trace_3* 
* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�_random_generator

Jkernel
Krecurrent_kernel
Lbias*
* 

H0
I1*

H0
I1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
B	variables
Ctrainable_variables
Dregularization_losses
F__call__
*G&call_and_return_all_conditional_losses
&G"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
\V
VARIABLE_VALUEdense/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUE
dense/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEgru/gru_cell/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEgru/gru_cell/recurrent_kernel&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
QK
VARIABLE_VALUEgru/gru_cell/bias&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
* 
C
0
1
2
3
4
5
6
7
	8*

�0
�1*
* 
* 
* 
* 
* 
* 
UO
VARIABLE_VALUEtraining/Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEtraining/Adam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEtraining/Adam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
WQ
VARIABLE_VALUEtraining/Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUEtraining/Adam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

@0*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

J0
K1
L2*

J0
K1
L2*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 
* 
* 
* 
* 
* 
* 
* 
M
�	variables
�	keras_api

�total

�count
�
_fn_kwargs*
z
�	variables
�	keras_api
�true_positives
�true_negatives
�false_positives
�false_negatives*
* 
* 
* 
* 
* 
* 
* 
* 
* 

�0
�1*

�	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*
* 
$
�0
�1
�2
�3*

�	variables*
e_
VARIABLE_VALUEtrue_positives=keras_api/metrics/1/true_positives/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUEtrue_negatives=keras_api/metrics/1/true_negatives/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUEfalse_positives>keras_api/metrics/1/false_positives/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUEfalse_negatives>keras_api/metrics/1/false_negatives/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEtraining/Adam/dense/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
�~
VARIABLE_VALUEtraining/Adam/dense/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE#training/Adam/gru/gru_cell/kernel/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE-training/Adam/gru/gru_cell/recurrent_kernel/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUE!training/Adam/gru/gru_cell/bias/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEtraining/Adam/dense/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
�~
VARIABLE_VALUEtraining/Adam/dense/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE#training/Adam/gru/gru_cell/kernel/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE-training/Adam/gru/gru_cell/recurrent_kernel/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUE!training/Adam/gru/gru_cell/bias/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenamedense/kernel
dense/biasgru/gru_cell/kernelgru/gru_cell/recurrent_kernelgru/gru_cell/biastraining/Adam/itertraining/Adam/beta_1training/Adam/beta_2training/Adam/decaytraining/Adam/learning_ratetotalcounttrue_positivestrue_negativesfalse_positivesfalse_negativestraining/Adam/dense/kernel/mtraining/Adam/dense/bias/m#training/Adam/gru/gru_cell/kernel/m-training/Adam/gru/gru_cell/recurrent_kernel/m!training/Adam/gru/gru_cell/bias/mtraining/Adam/dense/kernel/vtraining/Adam/dense/bias/v#training/Adam/gru/gru_cell/kernel/v-training/Adam/gru/gru_cell/recurrent_kernel/v!training/Adam/gru/gru_cell/bias/vConst*'
Tin 
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *&
f!R
__inference__traced_save_3571
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense/kernel
dense/biasgru/gru_cell/kernelgru/gru_cell/recurrent_kernelgru/gru_cell/biastraining/Adam/itertraining/Adam/beta_1training/Adam/beta_2training/Adam/decaytraining/Adam/learning_ratetotalcounttrue_positivestrue_negativesfalse_positivesfalse_negativestraining/Adam/dense/kernel/mtraining/Adam/dense/bias/m#training/Adam/gru/gru_cell/kernel/m-training/Adam/gru/gru_cell/recurrent_kernel/m!training/Adam/gru/gru_cell/bias/mtraining/Adam/dense/kernel/vtraining/Adam/dense/bias/v#training/Adam/gru/gru_cell/kernel/v-training/Adam/gru/gru_cell/recurrent_kernel/v!training/Adam/gru/gru_cell/bias/v*&
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *)
f$R"
 __inference__traced_restore_3658��
�
�
B__inference_gru_cell_layer_call_and_return_conditional_losses_1945

inputs

states2
 readvariableop_gru_gru_cell_bias:`;
)matmul_readvariableop_gru_gru_cell_kernel:`G
5matmul_1_readvariableop_gru_gru_cell_recurrent_kernel: `
identity

identity_1��MatMul/ReadVariableOp�MatMul_1/ReadVariableOp�ReadVariableOpo
ReadVariableOpReadVariableOp readvariableop_gru_gru_cell_bias*
_output_shapes

:`*
dtype0_
unstackUnpackReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
num
MatMul/ReadVariableOpReadVariableOp)matmul_readvariableop_gru_gru_cell_kernel*
_output_shapes

:`*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������`h
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*'
_output_shapes
:���������`Z
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*M
_output_shapes;
9:��������� :��������� :��������� *
	num_split�
MatMul_1/ReadVariableOpReadVariableOp5matmul_1_readvariableop_gru_gru_cell_recurrent_kernel*
_output_shapes

: `*
dtype0m
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������`l
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*'
_output_shapes
:���������`Z
ConstConst*
_output_shapes
:*
dtype0*!
valueB"        ����\
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
split_1SplitVBiasAdd_1:output:0Const:output:0split_1/split_dim:output:0*

Tlen0*
T0*M
_output_shapes;
9:��������� :��������� :��������� *
	num_split`
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:��������� M
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:��������� b
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:��������� Q
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:��������� ]
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:��������� Y
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:��������� I
TanhTanh	add_2:z:0*
T0*'
_output_shapes
:��������� S
mul_1MulSigmoid:y:0states*
T0*'
_output_shapes
:��������� J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:��������� Q
mul_2Mulsub:z:0Tanh:y:0*
T0*'
_output_shapes
:��������� V
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*'
_output_shapes
:��������� X
IdentityIdentity	add_3:z:0^NoOp*
T0*'
_output_shapes
:��������� Z

Identity_1Identity	add_3:z:0^NoOp*
T0*'
_output_shapes
:��������� e
NoOpNoOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
_output_shapes
 "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:���������:��������� : : : 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp:=9
7
_user_specified_namegru/gru_cell/recurrent_kernel:3/
-
_user_specified_namegru/gru_cell/kernel:1-
+
_user_specified_namegru/gru_cell/bias:OK
'
_output_shapes
:��������� 
 
_user_specified_namestates:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�9
�
while_body_2875
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
strided_slice_1_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0=
+gru_cell_readvariableop_gru_gru_cell_bias_0:`F
4gru_cell_matmul_readvariableop_gru_gru_cell_kernel_0:`R
@gru_cell_matmul_1_readvariableop_gru_gru_cell_recurrent_kernel_0: `
identity

identity_1

identity_2

identity_3

identity_4
strided_slice_1O
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor;
)gru_cell_readvariableop_gru_gru_cell_bias:`D
2gru_cell_matmul_readvariableop_gru_gru_cell_kernel:`P
>gru_cell_matmul_1_readvariableop_gru_gru_cell_recurrent_kernel: `��gru_cell/MatMul/ReadVariableOp� gru_cell/MatMul_1/ReadVariableOp�gru_cell/ReadVariableOp�
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
gru_cell/ReadVariableOpReadVariableOp+gru_cell_readvariableop_gru_gru_cell_bias_0*
_output_shapes

:`*
dtype0q
gru_cell/unstackUnpackgru_cell/ReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
num�
gru_cell/MatMul/ReadVariableOpReadVariableOp4gru_cell_matmul_readvariableop_gru_gru_cell_kernel_0*
_output_shapes

:`*
dtype0�
gru_cell/MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0&gru_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������`�
gru_cell/BiasAddBiasAddgru_cell/MatMul:product:0gru_cell/unstack:output:0*
T0*'
_output_shapes
:���������`c
gru_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
gru_cell/splitSplit!gru_cell/split/split_dim:output:0gru_cell/BiasAdd:output:0*
T0*M
_output_shapes;
9:��������� :��������� :��������� *
	num_split�
 gru_cell/MatMul_1/ReadVariableOpReadVariableOp@gru_cell_matmul_1_readvariableop_gru_gru_cell_recurrent_kernel_0*
_output_shapes

: `*
dtype0�
gru_cell/MatMul_1MatMulplaceholder_2(gru_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������`�
gru_cell/BiasAdd_1BiasAddgru_cell/MatMul_1:product:0gru_cell/unstack:output:1*
T0*'
_output_shapes
:���������`c
gru_cell/ConstConst*
_output_shapes
:*
dtype0*!
valueB"        ����e
gru_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
gru_cell/split_1SplitVgru_cell/BiasAdd_1:output:0gru_cell/Const:output:0#gru_cell/split_1/split_dim:output:0*

Tlen0*
T0*M
_output_shapes;
9:��������� :��������� :��������� *
	num_split{
gru_cell/addAddV2gru_cell/split:output:0gru_cell/split_1:output:0*
T0*'
_output_shapes
:��������� _
gru_cell/SigmoidSigmoidgru_cell/add:z:0*
T0*'
_output_shapes
:��������� }
gru_cell/add_1AddV2gru_cell/split:output:1gru_cell/split_1:output:1*
T0*'
_output_shapes
:��������� c
gru_cell/Sigmoid_1Sigmoidgru_cell/add_1:z:0*
T0*'
_output_shapes
:��������� x
gru_cell/mulMulgru_cell/Sigmoid_1:y:0gru_cell/split_1:output:2*
T0*'
_output_shapes
:��������� t
gru_cell/add_2AddV2gru_cell/split:output:2gru_cell/mul:z:0*
T0*'
_output_shapes
:��������� [
gru_cell/TanhTanhgru_cell/add_2:z:0*
T0*'
_output_shapes
:��������� l
gru_cell/mul_1Mulgru_cell/Sigmoid:y:0placeholder_2*
T0*'
_output_shapes
:��������� S
gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?t
gru_cell/subSubgru_cell/sub/x:output:0gru_cell/Sigmoid:y:0*
T0*'
_output_shapes
:��������� l
gru_cell/mul_2Mulgru_cell/sub:z:0gru_cell/Tanh:y:0*
T0*'
_output_shapes
:��������� q
gru_cell/add_3AddV2gru_cell/mul_1:z:0gru_cell/mul_2:z:0*
T0*'
_output_shapes
:��������� l
*TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : �
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_13TensorArrayV2Write/TensorListSetItem/index:output:0gru_cell/add_3:z:0*
_output_shapes
: *
element_dtype0:���G
add/yConst*
_output_shapes
: *
dtype0*
value	B :J
addAddV2placeholderadd/y:output:0*
T0*
_output_shapes
: I
add_1/yConst*
_output_shapes
: *
dtype0*
value	B :U
add_1AddV2while_loop_counteradd_1/y:output:0*
T0*
_output_shapes
: G
IdentityIdentity	add_1:z:0^NoOp*
T0*
_output_shapes
: X

Identity_1Identitywhile_maximum_iterations^NoOp*
T0*
_output_shapes
: G

Identity_2Identityadd:z:0^NoOp*
T0*
_output_shapes
: t

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^NoOp*
T0*
_output_shapes
: c

Identity_4Identitygru_cell/add_3:z:0^NoOp*
T0*'
_output_shapes
:��������� �
NoOpNoOp^gru_cell/MatMul/ReadVariableOp!^gru_cell/MatMul_1/ReadVariableOp^gru_cell/ReadVariableOp*
_output_shapes
 "�
>gru_cell_matmul_1_readvariableop_gru_gru_cell_recurrent_kernel@gru_cell_matmul_1_readvariableop_gru_gru_cell_recurrent_kernel_0"j
2gru_cell_matmul_readvariableop_gru_gru_cell_kernel4gru_cell_matmul_readvariableop_gru_gru_cell_kernel_0"X
)gru_cell_readvariableop_gru_gru_cell_bias+gru_cell_readvariableop_gru_gru_cell_bias_0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
identityIdentity:output:0"$
strided_slice_1strided_slice_1_0"�
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :��������� : : : : : 2@
gru_cell/MatMul/ReadVariableOpgru_cell/MatMul/ReadVariableOp2D
 gru_cell/MatMul_1/ReadVariableOp gru_cell/MatMul_1/ReadVariableOp22
gru_cell/ReadVariableOpgru_cell/ReadVariableOp:=	9
7
_user_specified_namegru/gru_cell/recurrent_kernel:3/
-
_user_specified_namegru/gru_cell/kernel:1-
+
_user_specified_namegru/gru_cell/bias:_[

_output_shapes
: 
A
_user_specified_name)'TensorArrayUnstack/TensorListFromTensor:GC

_output_shapes
: 
)
_user_specified_namestrided_slice_1:-)
'
_output_shapes
:��������� :

_output_shapes
: :

_output_shapes
: :PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter
�
P
$__inference_dot_1_layer_call_fn_2591
inputs_0
inputs_1
identity�
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *H
fCRA
?__inference_dot_1_layer_call_and_return_conditional_losses_2093e
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:����������
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:�����������:����������
:VR
,
_output_shapes
:����������

"
_user_specified_name
inputs_1:W S
-
_output_shapes
:�����������
"
_user_specified_name
inputs_0
�
o
E__inference_concatenate_layer_call_and_return_conditional_losses_2108

inputs
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :z
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*,
_output_shapes
:����������\
IdentityIdentityconcat:output:0*
T0*,
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:����������
:����������
:TP
,
_output_shapes
:����������

 
_user_specified_nameinputs:T P
,
_output_shapes
:����������

 
_user_specified_nameinputs
�
H
,__inference_masking_prior_layer_call_fn_2549

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_masking_prior_layer_call_and_return_conditional_losses_2069e
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:����������
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������
:T P
,
_output_shapes
:����������

 
_user_specified_nameinputs
�
`
D__inference_activation_layer_call_and_return_conditional_losses_2585

inputs
identityR
SoftmaxSoftmaxinputs*
T0*-
_output_shapes
:�����������_
IdentityIdentitySoftmax:softmax:0*
T0*-
_output_shapes
:�����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:�����������:U Q
-
_output_shapes
:�����������
 
_user_specified_nameinputs
�
�
while_cond_3029
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
less_strided_slice_1,
(while_cond_3029___redundant_placeholder0,
(while_cond_3029___redundant_placeholder1,
(while_cond_3029___redundant_placeholder2,
(while_cond_3029___redundant_placeholder3
identity
P
LessLessplaceholderless_strided_slice_1*
T0*
_output_shapes
: ?
IdentityIdentityLess:z:0*
T0
*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :��������� : :::::

_output_shapes
::GC

_output_shapes
: 
)
_user_specified_namestrided_slice_1:-)
'
_output_shapes
:��������� :

_output_shapes
: :

_output_shapes
: :PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter
�
E
)__inference_activation_layer_call_fn_2580

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:�����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_activation_layer_call_and_return_conditional_losses_2085f
IdentityIdentityPartitionedCall:output:0*
T0*-
_output_shapes
:�����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:�����������:U Q
-
_output_shapes
:�����������
 
_user_specified_nameinputs
�
�
'__inference_gru_cell_layer_call_fn_3315

inputs
states_0#
gru_gru_cell_bias:`%
gru_gru_cell_kernel:`/
gru_gru_cell_recurrent_kernel: `
identity

identity_1��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0gru_gru_cell_biasgru_gru_cell_kernelgru_gru_cell_recurrent_kernel*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:��������� :��������� *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_gru_cell_layer_call_and_return_conditional_losses_1945o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:��������� q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:��������� <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:���������:��������� : : : 22
StatefulPartitionedCallStatefulPartitionedCall:=9
7
_user_specified_namegru/gru_cell/recurrent_kernel:3/
-
_user_specified_namegru/gru_cell/kernel:1-
+
_user_specified_namegru/gru_cell/bias:QM
'
_output_shapes
:��������� 
"
_user_specified_name
states_0:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
S
'__inference_multiply_layer_call_fn_2604
inputs_0
inputs_1
identity�
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_multiply_layer_call_and_return_conditional_losses_2100e
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:����������
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:����������
:����������
:VR
,
_output_shapes
:����������

"
_user_specified_name
inputs_1:V R
,
_output_shapes
:����������

"
_user_specified_name
inputs_0
�9
�
while_body_3030
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
strided_slice_1_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0=
+gru_cell_readvariableop_gru_gru_cell_bias_0:`F
4gru_cell_matmul_readvariableop_gru_gru_cell_kernel_0:`R
@gru_cell_matmul_1_readvariableop_gru_gru_cell_recurrent_kernel_0: `
identity

identity_1

identity_2

identity_3

identity_4
strided_slice_1O
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor;
)gru_cell_readvariableop_gru_gru_cell_bias:`D
2gru_cell_matmul_readvariableop_gru_gru_cell_kernel:`P
>gru_cell_matmul_1_readvariableop_gru_gru_cell_recurrent_kernel: `��gru_cell/MatMul/ReadVariableOp� gru_cell/MatMul_1/ReadVariableOp�gru_cell/ReadVariableOp�
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
gru_cell/ReadVariableOpReadVariableOp+gru_cell_readvariableop_gru_gru_cell_bias_0*
_output_shapes

:`*
dtype0q
gru_cell/unstackUnpackgru_cell/ReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
num�
gru_cell/MatMul/ReadVariableOpReadVariableOp4gru_cell_matmul_readvariableop_gru_gru_cell_kernel_0*
_output_shapes

:`*
dtype0�
gru_cell/MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0&gru_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������`�
gru_cell/BiasAddBiasAddgru_cell/MatMul:product:0gru_cell/unstack:output:0*
T0*'
_output_shapes
:���������`c
gru_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
gru_cell/splitSplit!gru_cell/split/split_dim:output:0gru_cell/BiasAdd:output:0*
T0*M
_output_shapes;
9:��������� :��������� :��������� *
	num_split�
 gru_cell/MatMul_1/ReadVariableOpReadVariableOp@gru_cell_matmul_1_readvariableop_gru_gru_cell_recurrent_kernel_0*
_output_shapes

: `*
dtype0�
gru_cell/MatMul_1MatMulplaceholder_2(gru_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������`�
gru_cell/BiasAdd_1BiasAddgru_cell/MatMul_1:product:0gru_cell/unstack:output:1*
T0*'
_output_shapes
:���������`c
gru_cell/ConstConst*
_output_shapes
:*
dtype0*!
valueB"        ����e
gru_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
gru_cell/split_1SplitVgru_cell/BiasAdd_1:output:0gru_cell/Const:output:0#gru_cell/split_1/split_dim:output:0*

Tlen0*
T0*M
_output_shapes;
9:��������� :��������� :��������� *
	num_split{
gru_cell/addAddV2gru_cell/split:output:0gru_cell/split_1:output:0*
T0*'
_output_shapes
:��������� _
gru_cell/SigmoidSigmoidgru_cell/add:z:0*
T0*'
_output_shapes
:��������� }
gru_cell/add_1AddV2gru_cell/split:output:1gru_cell/split_1:output:1*
T0*'
_output_shapes
:��������� c
gru_cell/Sigmoid_1Sigmoidgru_cell/add_1:z:0*
T0*'
_output_shapes
:��������� x
gru_cell/mulMulgru_cell/Sigmoid_1:y:0gru_cell/split_1:output:2*
T0*'
_output_shapes
:��������� t
gru_cell/add_2AddV2gru_cell/split:output:2gru_cell/mul:z:0*
T0*'
_output_shapes
:��������� [
gru_cell/TanhTanhgru_cell/add_2:z:0*
T0*'
_output_shapes
:��������� l
gru_cell/mul_1Mulgru_cell/Sigmoid:y:0placeholder_2*
T0*'
_output_shapes
:��������� S
gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?t
gru_cell/subSubgru_cell/sub/x:output:0gru_cell/Sigmoid:y:0*
T0*'
_output_shapes
:��������� l
gru_cell/mul_2Mulgru_cell/sub:z:0gru_cell/Tanh:y:0*
T0*'
_output_shapes
:��������� q
gru_cell/add_3AddV2gru_cell/mul_1:z:0gru_cell/mul_2:z:0*
T0*'
_output_shapes
:��������� l
*TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : �
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_13TensorArrayV2Write/TensorListSetItem/index:output:0gru_cell/add_3:z:0*
_output_shapes
: *
element_dtype0:���G
add/yConst*
_output_shapes
: *
dtype0*
value	B :J
addAddV2placeholderadd/y:output:0*
T0*
_output_shapes
: I
add_1/yConst*
_output_shapes
: *
dtype0*
value	B :U
add_1AddV2while_loop_counteradd_1/y:output:0*
T0*
_output_shapes
: G
IdentityIdentity	add_1:z:0^NoOp*
T0*
_output_shapes
: X

Identity_1Identitywhile_maximum_iterations^NoOp*
T0*
_output_shapes
: G

Identity_2Identityadd:z:0^NoOp*
T0*
_output_shapes
: t

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^NoOp*
T0*
_output_shapes
: c

Identity_4Identitygru_cell/add_3:z:0^NoOp*
T0*'
_output_shapes
:��������� �
NoOpNoOp^gru_cell/MatMul/ReadVariableOp!^gru_cell/MatMul_1/ReadVariableOp^gru_cell/ReadVariableOp*
_output_shapes
 "�
>gru_cell_matmul_1_readvariableop_gru_gru_cell_recurrent_kernel@gru_cell_matmul_1_readvariableop_gru_gru_cell_recurrent_kernel_0"j
2gru_cell_matmul_readvariableop_gru_gru_cell_kernel4gru_cell_matmul_readvariableop_gru_gru_cell_kernel_0"X
)gru_cell_readvariableop_gru_gru_cell_bias+gru_cell_readvariableop_gru_gru_cell_bias_0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
identityIdentity:output:0"$
strided_slice_1strided_slice_1_0"�
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :��������� : : : : : 2@
gru_cell/MatMul/ReadVariableOpgru_cell/MatMul/ReadVariableOp2D
 gru_cell/MatMul_1/ReadVariableOp gru_cell/MatMul_1/ReadVariableOp22
gru_cell/ReadVariableOpgru_cell/ReadVariableOp:=	9
7
_user_specified_namegru/gru_cell/recurrent_kernel:3/
-
_user_specified_namegru/gru_cell/kernel:1-
+
_user_specified_namegru/gru_cell/bias:_[

_output_shapes
: 
A
_user_specified_name)'TensorArrayUnstack/TensorListFromTensor:GC

_output_shapes
: 
)
_user_specified_namestrided_slice_1:-)
'
_output_shapes
:��������� :

_output_shapes
: :

_output_shapes
: :PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter
��
�
__inference__wrapped_model_1742
input_priorE
3model_gru_gru_cell_readvariableop_gru_gru_cell_bias:`N
<model_gru_gru_cell_matmul_readvariableop_gru_gru_cell_kernel:`Z
Hmodel_gru_gru_cell_matmul_1_readvariableop_gru_gru_cell_recurrent_kernel: `@
.model_dense_matmul_readvariableop_dense_kernel: ;
-model_dense_biasadd_readvariableop_dense_bias:
identity��"model/dense/BiasAdd/ReadVariableOp�!model/dense/MatMul/ReadVariableOp�(model/gru/gru_cell/MatMul/ReadVariableOp�*model/gru/gru_cell/MatMul_1/ReadVariableOp�!model/gru/gru_cell/ReadVariableOp�model/gru/whilec
model/masking_prior/NotEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ���
model/masking_prior/NotEqualNotEqualinput_prior'model/masking_prior/NotEqual/y:output:0*
T0*,
_output_shapes
:����������
t
)model/masking_prior/Any/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
����������
model/masking_prior/AnyAny model/masking_prior/NotEqual:z:02model/masking_prior/Any/reduction_indices:output:0*,
_output_shapes
:����������*
	keep_dims(�
model/masking_prior/CastCast model/masking_prior/Any:output:0*

DstT0*

SrcT0
*,
_output_shapes
:�����������
model/masking_prior/mulMulinput_priormodel/masking_prior/Cast:y:0*
T0*,
_output_shapes
:����������
�
model/masking_prior/SqueezeSqueeze model/masking_prior/Any:output:0*
T0
*(
_output_shapes
:����������*
squeeze_dims

���������m
model/dot/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
model/dot/transpose	Transposemodel/masking_prior/mul:z:0!model/dot/transpose/perm:output:0*
T0*,
_output_shapes
:���������
��
model/dot/MatMulBatchMatMulV2model/masking_prior/mul:z:0model/dot/transpose:y:0*
T0*-
_output_shapes
:�����������f
model/dot/ShapeShapemodel/dot/MatMul:output:0*
T0*
_output_shapes
::��v
model/activation/SoftmaxSoftmaxmodel/dot/MatMul:output:0*
T0*-
_output_shapes
:������������
model/dot_1/MatMulBatchMatMulV2"model/activation/Softmax:softmax:0model/masking_prior/mul:z:0*
T0*,
_output_shapes
:����������
j
model/dot_1/ShapeShapemodel/dot_1/MatMul:output:0*
T0*
_output_shapes
::���
model/multiply/mulMulmodel/dot_1/MatMul:output:0model/masking_prior/mul:z:0*
T0*,
_output_shapes
:����������
_
model/multiply/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : �
model/multiply/ExpandDims
ExpandDims$model/masking_prior/Squeeze:output:0&model/multiply/ExpandDims/dim:output:0*
T0
*,
_output_shapes
:����������b
 model/multiply/concat/concat_dimConst*
_output_shapes
: *
dtype0*
value	B : �
model/multiply/concat/concatIdentity"model/multiply/ExpandDims:output:0*
T0
*,
_output_shapes
:����������f
$model/multiply/All/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : �
model/multiply/AllAll%model/multiply/concat/concat:output:0-model/multiply/All/reduction_indices:output:0*(
_output_shapes
:����������_
model/concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
model/concatenate/concatConcatV2model/masking_prior/mul:z:0model/multiply/mul:z:0&model/concatenate/concat/axis:output:0*
N*
T0*,
_output_shapes
:����������k
 model/concatenate/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
model/concatenate/ExpandDims
ExpandDims$model/masking_prior/Squeeze:output:0)model/concatenate/ExpandDims/dim:output:0*
T0
*,
_output_shapes
:����������m
"model/concatenate/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
model/concatenate/ExpandDims_1
ExpandDimsmodel/multiply/All:output:0+model/concatenate/ExpandDims_1/dim:output:0*
T0
*,
_output_shapes
:����������a
model/concatenate/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B :�
model/concatenate/concat_1ConcatV2%model/concatenate/ExpandDims:output:0'model/concatenate/ExpandDims_1:output:0(model/concatenate/concat_1/axis:output:0*
N*
T0
*,
_output_shapes
:����������r
'model/concatenate/All/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
����������
model/concatenate/AllAll#model/concatenate/concat_1:output:00model/concatenate/All/reduction_indices:output:0*(
_output_shapes
:����������n
model/gru/ShapeShape!model/concatenate/concat:output:0*
T0*
_output_shapes
::��g
model/gru/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: i
model/gru/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
model/gru/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
model/gru/strided_sliceStridedSlicemodel/gru/Shape:output:0&model/gru/strided_slice/stack:output:0(model/gru/strided_slice/stack_1:output:0(model/gru/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskZ
model/gru/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : �
model/gru/zeros/packedPack model/gru/strided_slice:output:0!model/gru/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:Z
model/gru/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
model/gru/zerosFillmodel/gru/zeros/packed:output:0model/gru/zeros/Const:output:0*
T0*'
_output_shapes
:��������� m
model/gru/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
model/gru/transpose	Transpose!model/concatenate/concat:output:0!model/gru/transpose/perm:output:0*
T0*,
_output_shapes
:����������f
model/gru/Shape_1Shapemodel/gru/transpose:y:0*
T0*
_output_shapes
::��i
model/gru/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: k
!model/gru/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:k
!model/gru/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
model/gru/strided_slice_1StridedSlicemodel/gru/Shape_1:output:0(model/gru/strided_slice_1/stack:output:0*model/gru/strided_slice_1/stack_1:output:0*model/gru/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskc
model/gru/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
model/gru/ExpandDims
ExpandDimsmodel/concatenate/All:output:0!model/gru/ExpandDims/dim:output:0*
T0
*,
_output_shapes
:����������o
model/gru/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
model/gru/transpose_1	Transposemodel/gru/ExpandDims:output:0#model/gru/transpose_1/perm:output:0*
T0
*,
_output_shapes
:����������p
%model/gru/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
model/gru/TensorArrayV2TensorListReserve.model/gru/TensorArrayV2/element_shape:output:0"model/gru/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
?model/gru/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
1model/gru/TensorArrayUnstack/TensorListFromTensorTensorListFromTensormodel/gru/transpose:y:0Hmodel/gru/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���i
model/gru/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: k
!model/gru/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:k
!model/gru/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
model/gru/strided_slice_2StridedSlicemodel/gru/transpose:y:0(model/gru/strided_slice_2/stack:output:0*model/gru/strided_slice_2/stack_1:output:0*model/gru/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask�
!model/gru/gru_cell/ReadVariableOpReadVariableOp3model_gru_gru_cell_readvariableop_gru_gru_cell_bias*
_output_shapes

:`*
dtype0�
model/gru/gru_cell/unstackUnpack)model/gru/gru_cell/ReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
num�
(model/gru/gru_cell/MatMul/ReadVariableOpReadVariableOp<model_gru_gru_cell_matmul_readvariableop_gru_gru_cell_kernel*
_output_shapes

:`*
dtype0�
model/gru/gru_cell/MatMulMatMul"model/gru/strided_slice_2:output:00model/gru/gru_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������`�
model/gru/gru_cell/BiasAddBiasAdd#model/gru/gru_cell/MatMul:product:0#model/gru/gru_cell/unstack:output:0*
T0*'
_output_shapes
:���������`m
"model/gru/gru_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
model/gru/gru_cell/splitSplit+model/gru/gru_cell/split/split_dim:output:0#model/gru/gru_cell/BiasAdd:output:0*
T0*M
_output_shapes;
9:��������� :��������� :��������� *
	num_split�
*model/gru/gru_cell/MatMul_1/ReadVariableOpReadVariableOpHmodel_gru_gru_cell_matmul_1_readvariableop_gru_gru_cell_recurrent_kernel*
_output_shapes

: `*
dtype0�
model/gru/gru_cell/MatMul_1MatMulmodel/gru/zeros:output:02model/gru/gru_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������`�
model/gru/gru_cell/BiasAdd_1BiasAdd%model/gru/gru_cell/MatMul_1:product:0#model/gru/gru_cell/unstack:output:1*
T0*'
_output_shapes
:���������`m
model/gru/gru_cell/ConstConst*
_output_shapes
:*
dtype0*!
valueB"        ����o
$model/gru/gru_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
model/gru/gru_cell/split_1SplitV%model/gru/gru_cell/BiasAdd_1:output:0!model/gru/gru_cell/Const:output:0-model/gru/gru_cell/split_1/split_dim:output:0*

Tlen0*
T0*M
_output_shapes;
9:��������� :��������� :��������� *
	num_split�
model/gru/gru_cell/addAddV2!model/gru/gru_cell/split:output:0#model/gru/gru_cell/split_1:output:0*
T0*'
_output_shapes
:��������� s
model/gru/gru_cell/SigmoidSigmoidmodel/gru/gru_cell/add:z:0*
T0*'
_output_shapes
:��������� �
model/gru/gru_cell/add_1AddV2!model/gru/gru_cell/split:output:1#model/gru/gru_cell/split_1:output:1*
T0*'
_output_shapes
:��������� w
model/gru/gru_cell/Sigmoid_1Sigmoidmodel/gru/gru_cell/add_1:z:0*
T0*'
_output_shapes
:��������� �
model/gru/gru_cell/mulMul model/gru/gru_cell/Sigmoid_1:y:0#model/gru/gru_cell/split_1:output:2*
T0*'
_output_shapes
:��������� �
model/gru/gru_cell/add_2AddV2!model/gru/gru_cell/split:output:2model/gru/gru_cell/mul:z:0*
T0*'
_output_shapes
:��������� o
model/gru/gru_cell/TanhTanhmodel/gru/gru_cell/add_2:z:0*
T0*'
_output_shapes
:��������� �
model/gru/gru_cell/mul_1Mulmodel/gru/gru_cell/Sigmoid:y:0model/gru/zeros:output:0*
T0*'
_output_shapes
:��������� ]
model/gru/gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
model/gru/gru_cell/subSub!model/gru/gru_cell/sub/x:output:0model/gru/gru_cell/Sigmoid:y:0*
T0*'
_output_shapes
:��������� �
model/gru/gru_cell/mul_2Mulmodel/gru/gru_cell/sub:z:0model/gru/gru_cell/Tanh:y:0*
T0*'
_output_shapes
:��������� �
model/gru/gru_cell/add_3AddV2model/gru/gru_cell/mul_1:z:0model/gru/gru_cell/mul_2:z:0*
T0*'
_output_shapes
:��������� x
'model/gru/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    h
&model/gru/TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :�
model/gru/TensorArrayV2_1TensorListReserve0model/gru/TensorArrayV2_1/element_shape:output:0/model/gru/TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���P
model/gru/timeConst*
_output_shapes
: *
dtype0*
value	B : r
'model/gru/TensorArrayV2_2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
model/gru/TensorArrayV2_2TensorListReserve0model/gru/TensorArrayV2_2/element_shape:output:0"model/gru/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0
*

shape_type0:����
Amodel/gru/TensorArrayUnstack_1/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
3model/gru/TensorArrayUnstack_1/TensorListFromTensorTensorListFromTensormodel/gru/transpose_1:y:0Jmodel/gru/TensorArrayUnstack_1/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0
*

shape_type0:���q
model/gru/zeros_like	ZerosLikemodel/gru/gru_cell/add_3:z:0*
T0*'
_output_shapes
:��������� m
"model/gru/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������^
model/gru/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
model/gru/whileWhile%model/gru/while/loop_counter:output:0+model/gru/while/maximum_iterations:output:0model/gru/time:output:0"model/gru/TensorArrayV2_1:handle:0model/gru/zeros_like:y:0model/gru/zeros:output:0"model/gru/strided_slice_1:output:0Amodel/gru/TensorArrayUnstack/TensorListFromTensor:output_handle:0Cmodel/gru/TensorArrayUnstack_1/TensorListFromTensor:output_handle:03model_gru_gru_cell_readvariableop_gru_gru_cell_bias<model_gru_gru_cell_matmul_readvariableop_gru_gru_cell_kernelHmodel_gru_gru_cell_matmul_1_readvariableop_gru_gru_cell_recurrent_kernel*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :��������� :��������� : : : : : : *%
_read_only_resource_inputs
	
*%
bodyR
model_gru_while_body_1630*%
condR
model_gru_while_cond_1629*M
output_shapes<
:: : : : :��������� :��������� : : : : : : *
parallel_iterations �
:model/gru/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    �
,model/gru/TensorArrayV2Stack/TensorListStackTensorListStackmodel/gru/while:output:3Cmodel/gru/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:��������� *
element_dtype0*
num_elementsr
model/gru/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������k
!model/gru/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: k
!model/gru/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
model/gru/strided_slice_3StridedSlice5model/gru/TensorArrayV2Stack/TensorListStack:tensor:0(model/gru/strided_slice_3/stack:output:0*model/gru/strided_slice_3/stack_1:output:0*model/gru/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:��������� *
shrink_axis_masko
model/gru/transpose_2/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
model/gru/transpose_2	Transpose5model/gru/TensorArrayV2Stack/TensorListStack:tensor:0#model/gru/transpose_2/perm:output:0*
T0*+
_output_shapes
:��������� e
model/gru/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    �
!model/dense/MatMul/ReadVariableOpReadVariableOp.model_dense_matmul_readvariableop_dense_kernel*
_output_shapes

: *
dtype0�
model/dense/MatMulMatMul"model/gru/strided_slice_3:output:0)model/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
"model/dense/BiasAdd/ReadVariableOpReadVariableOp-model_dense_biasadd_readvariableop_dense_bias*
_output_shapes
:*
dtype0�
model/dense/BiasAddBiasAddmodel/dense/MatMul:product:0*model/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������n
model/dense/SoftmaxSoftmaxmodel/dense/BiasAdd:output:0*
T0*'
_output_shapes
:���������l
IdentityIdentitymodel/dense/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp#^model/dense/BiasAdd/ReadVariableOp"^model/dense/MatMul/ReadVariableOp)^model/gru/gru_cell/MatMul/ReadVariableOp+^model/gru/gru_cell/MatMul_1/ReadVariableOp"^model/gru/gru_cell/ReadVariableOp^model/gru/while*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":����������
: : : : : 2H
"model/dense/BiasAdd/ReadVariableOp"model/dense/BiasAdd/ReadVariableOp2F
!model/dense/MatMul/ReadVariableOp!model/dense/MatMul/ReadVariableOp2T
(model/gru/gru_cell/MatMul/ReadVariableOp(model/gru/gru_cell/MatMul/ReadVariableOp2X
*model/gru/gru_cell/MatMul_1/ReadVariableOp*model/gru/gru_cell/MatMul_1/ReadVariableOp2F
!model/gru/gru_cell/ReadVariableOp!model/gru/gru_cell/ReadVariableOp2"
model/gru/whilemodel/gru/while:*&
$
_user_specified_name
dense/bias:,(
&
_user_specified_namedense/kernel:=9
7
_user_specified_namegru/gru_cell/recurrent_kernel:3/
-
_user_specified_namegru/gru_cell/kernel:1-
+
_user_specified_namegru/gru_cell/bias:Y U
,
_output_shapes
:����������

%
_user_specified_nameinput_prior
�
n
B__inference_multiply_layer_call_and_return_conditional_losses_2610
inputs_0
inputs_1
identityU
mulMulinputs_0inputs_1*
T0*,
_output_shapes
:����������
T
IdentityIdentitymul:z:0*
T0*,
_output_shapes
:����������
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:����������
:����������
:VR
,
_output_shapes
:����������

"
_user_specified_name
inputs_1:V R
,
_output_shapes
:����������

"
_user_specified_name
inputs_0
�6
�
=__inference_gru_layer_call_and_return_conditional_losses_2018

inputs,
gru_cell_gru_gru_cell_bias:`.
gru_cell_gru_gru_cell_kernel:`8
&gru_cell_gru_gru_cell_recurrent_kernel: `
identity�� gru_cell/StatefulPartitionedCall�whileI
ShapeShapeinputs*
T0*
_output_shapes
::��]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:��������� c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :������������������R
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
::��_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask�
 gru_cell/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0gru_cell_gru_gru_cell_biasgru_cell_gru_gru_cell_kernel&gru_cell_gru_gru_cell_recurrent_kernel*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:��������� :��������� *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_gru_cell_layer_call_and_return_conditional_losses_1945n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :�
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0gru_cell_gru_gru_cell_biasgru_cell_gru_gru_cell_kernel&gru_cell_gru_gru_cell_recurrent_kernel*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :��������� : : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_1956*
condR
while_cond_1955*8
output_shapes'
%: : : : :��������� : : : : : *
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:��������� *
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:��������� *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:��������� [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:��������� M
NoOpNoOp!^gru_cell/StatefulPartitionedCall^while*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������: : : 2D
 gru_cell/StatefulPartitionedCall gru_cell/StatefulPartitionedCall2
whilewhile:=9
7
_user_specified_namegru/gru_cell/recurrent_kernel:3/
-
_user_specified_namegru/gru_cell/kernel:1-
+
_user_specified_namegru/gru_cell/bias:\ X
4
_output_shapes"
 :������������������
 
_user_specified_nameinputs
�
�
B__inference_gru_cell_layer_call_and_return_conditional_losses_3354

inputs
states_02
 readvariableop_gru_gru_cell_bias:`;
)matmul_readvariableop_gru_gru_cell_kernel:`G
5matmul_1_readvariableop_gru_gru_cell_recurrent_kernel: `
identity

identity_1��MatMul/ReadVariableOp�MatMul_1/ReadVariableOp�ReadVariableOpo
ReadVariableOpReadVariableOp readvariableop_gru_gru_cell_bias*
_output_shapes

:`*
dtype0_
unstackUnpackReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
num
MatMul/ReadVariableOpReadVariableOp)matmul_readvariableop_gru_gru_cell_kernel*
_output_shapes

:`*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������`h
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*'
_output_shapes
:���������`Z
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*M
_output_shapes;
9:��������� :��������� :��������� *
	num_split�
MatMul_1/ReadVariableOpReadVariableOp5matmul_1_readvariableop_gru_gru_cell_recurrent_kernel*
_output_shapes

: `*
dtype0o
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������`l
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*'
_output_shapes
:���������`Z
ConstConst*
_output_shapes
:*
dtype0*!
valueB"        ����\
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
split_1SplitVBiasAdd_1:output:0Const:output:0split_1/split_dim:output:0*

Tlen0*
T0*M
_output_shapes;
9:��������� :��������� :��������� *
	num_split`
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:��������� M
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:��������� b
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:��������� Q
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:��������� ]
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:��������� Y
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:��������� I
TanhTanh	add_2:z:0*
T0*'
_output_shapes
:��������� U
mul_1MulSigmoid:y:0states_0*
T0*'
_output_shapes
:��������� J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:��������� Q
mul_2Mulsub:z:0Tanh:y:0*
T0*'
_output_shapes
:��������� V
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*'
_output_shapes
:��������� X
IdentityIdentity	add_3:z:0^NoOp*
T0*'
_output_shapes
:��������� Z

Identity_1Identity	add_3:z:0^NoOp*
T0*'
_output_shapes
:��������� e
NoOpNoOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
_output_shapes
 "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:���������:��������� : : : 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp:=9
7
_user_specified_namegru/gru_cell/recurrent_kernel:3/
-
_user_specified_namegru/gru_cell/kernel:1-
+
_user_specified_namegru/gru_cell/bias:QM
'
_output_shapes
:��������� 
"
_user_specified_name
states_0:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�

�
?__inference_dense_layer_call_and_return_conditional_losses_2279

inputs4
"matmul_readvariableop_dense_kernel: /
!biasadd_readvariableop_dense_bias:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpx
MatMul/ReadVariableOpReadVariableOp"matmul_readvariableop_dense_kernel*
_output_shapes

: *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������t
BiasAdd/ReadVariableOpReadVariableOp!biasadd_readvariableop_dense_bias*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:���������`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:���������S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:��������� : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:*&
$
_user_specified_name
dense/bias:,(
&
_user_specified_namedense/kernel:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
l
B__inference_multiply_layer_call_and_return_conditional_losses_2100

inputs
inputs_1
identityS
mulMulinputsinputs_1*
T0*,
_output_shapes
:����������
T
IdentityIdentitymul:z:0*
T0*,
_output_shapes
:����������
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:����������
:����������
:TP
,
_output_shapes
:����������

 
_user_specified_nameinputs:T P
,
_output_shapes
:����������

 
_user_specified_nameinputs
�
�
B__inference_gru_cell_layer_call_and_return_conditional_losses_1807

inputs

states2
 readvariableop_gru_gru_cell_bias:`;
)matmul_readvariableop_gru_gru_cell_kernel:`G
5matmul_1_readvariableop_gru_gru_cell_recurrent_kernel: `
identity

identity_1��MatMul/ReadVariableOp�MatMul_1/ReadVariableOp�ReadVariableOpo
ReadVariableOpReadVariableOp readvariableop_gru_gru_cell_bias*
_output_shapes

:`*
dtype0_
unstackUnpackReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
num
MatMul/ReadVariableOpReadVariableOp)matmul_readvariableop_gru_gru_cell_kernel*
_output_shapes

:`*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������`h
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*'
_output_shapes
:���������`Z
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*M
_output_shapes;
9:��������� :��������� :��������� *
	num_split�
MatMul_1/ReadVariableOpReadVariableOp5matmul_1_readvariableop_gru_gru_cell_recurrent_kernel*
_output_shapes

: `*
dtype0m
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������`l
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*'
_output_shapes
:���������`Z
ConstConst*
_output_shapes
:*
dtype0*!
valueB"        ����\
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
split_1SplitVBiasAdd_1:output:0Const:output:0split_1/split_dim:output:0*

Tlen0*
T0*M
_output_shapes;
9:��������� :��������� :��������� *
	num_split`
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:��������� M
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:��������� b
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:��������� Q
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:��������� ]
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:��������� Y
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:��������� I
TanhTanh	add_2:z:0*
T0*'
_output_shapes
:��������� S
mul_1MulSigmoid:y:0states*
T0*'
_output_shapes
:��������� J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:��������� Q
mul_2Mulsub:z:0Tanh:y:0*
T0*'
_output_shapes
:��������� V
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*'
_output_shapes
:��������� X
IdentityIdentity	add_3:z:0^NoOp*
T0*'
_output_shapes
:��������� Z

Identity_1Identity	add_3:z:0^NoOp*
T0*'
_output_shapes
:��������� e
NoOpNoOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
_output_shapes
 "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:���������:��������� : : : 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp:=9
7
_user_specified_namegru/gru_cell/recurrent_kernel:3/
-
_user_specified_namegru/gru_cell/kernel:1-
+
_user_specified_namegru/gru_cell/bias:OK
'
_output_shapes
:��������� 
 
_user_specified_namestates:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�	
�
"__inference_gru_layer_call_fn_2655

inputs#
gru_gru_cell_bias:`%
gru_gru_cell_kernel:`/
gru_gru_cell_recurrent_kernel: `
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsgru_gru_cell_biasgru_gru_cell_kernelgru_gru_cell_recurrent_kernel*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *F
fAR?
=__inference_gru_layer_call_and_return_conditional_losses_2447o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:��������� <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:����������: : : 22
StatefulPartitionedCallStatefulPartitionedCall:=9
7
_user_specified_namegru/gru_cell/recurrent_kernel:3/
-
_user_specified_namegru/gru_cell/kernel:1-
+
_user_specified_namegru/gru_cell/bias:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�!
�
while_body_1956
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
strided_slice_1_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0.
gru_cell_gru_gru_cell_bias_0:`0
gru_cell_gru_gru_cell_kernel_0:`:
(gru_cell_gru_gru_cell_recurrent_kernel_0: `
identity

identity_1

identity_2

identity_3

identity_4
strided_slice_1O
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor,
gru_cell_gru_gru_cell_bias:`.
gru_cell_gru_gru_cell_kernel:`8
&gru_cell_gru_gru_cell_recurrent_kernel: `�� gru_cell/StatefulPartitionedCall�
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
 gru_cell/StatefulPartitionedCallStatefulPartitionedCall*TensorArrayV2Read/TensorListGetItem:item:0placeholder_2gru_cell_gru_gru_cell_bias_0gru_cell_gru_gru_cell_kernel_0(gru_cell_gru_gru_cell_recurrent_kernel_0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:��������� :��������� *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_gru_cell_layer_call_and_return_conditional_losses_1945l
*TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : �
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_13TensorArrayV2Write/TensorListSetItem/index:output:0)gru_cell/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:���G
add/yConst*
_output_shapes
: *
dtype0*
value	B :J
addAddV2placeholderadd/y:output:0*
T0*
_output_shapes
: I
add_1/yConst*
_output_shapes
: *
dtype0*
value	B :U
add_1AddV2while_loop_counteradd_1/y:output:0*
T0*
_output_shapes
: G
IdentityIdentity	add_1:z:0^NoOp*
T0*
_output_shapes
: X

Identity_1Identitywhile_maximum_iterations^NoOp*
T0*
_output_shapes
: G

Identity_2Identityadd:z:0^NoOp*
T0*
_output_shapes
: t

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^NoOp*
T0*
_output_shapes
: z

Identity_4Identity)gru_cell/StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:��������� E
NoOpNoOp!^gru_cell/StatefulPartitionedCall*
_output_shapes
 ":
gru_cell_gru_gru_cell_biasgru_cell_gru_gru_cell_bias_0">
gru_cell_gru_gru_cell_kernelgru_cell_gru_gru_cell_kernel_0"R
&gru_cell_gru_gru_cell_recurrent_kernel(gru_cell_gru_gru_cell_recurrent_kernel_0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
identityIdentity:output:0"$
strided_slice_1strided_slice_1_0"�
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :��������� : : : : : 2D
 gru_cell/StatefulPartitionedCall gru_cell/StatefulPartitionedCall:=	9
7
_user_specified_namegru/gru_cell/recurrent_kernel:3/
-
_user_specified_namegru/gru_cell/kernel:1-
+
_user_specified_namegru/gru_cell/bias:_[

_output_shapes
: 
A
_user_specified_name)'TensorArrayUnstack/TensorListFromTensor:GC

_output_shapes
: 
)
_user_specified_namestrided_slice_1:-)
'
_output_shapes
:��������� :

_output_shapes
: :

_output_shapes
: :PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter
�
i
?__inference_dot_1_layer_call_and_return_conditional_losses_2093

inputs
inputs_1
identity`
MatMulBatchMatMulV2inputsinputs_1*
T0*,
_output_shapes
:����������
R
ShapeShapeMatMul:output:0*
T0*
_output_shapes
::��\
IdentityIdentityMatMul:output:0*
T0*,
_output_shapes
:����������
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:�����������:����������
:TP
,
_output_shapes
:����������

 
_user_specified_nameinputs:U Q
-
_output_shapes
:�����������
 
_user_specified_nameinputs
�	
�
"__inference_gru_layer_call_fn_2631
inputs_0#
gru_gru_cell_bias:`%
gru_gru_cell_kernel:`/
gru_gru_cell_recurrent_kernel: `
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0gru_gru_cell_biasgru_gru_cell_kernelgru_gru_cell_recurrent_kernel*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *F
fAR?
=__inference_gru_layer_call_and_return_conditional_losses_1880o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:��������� <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������: : : 22
StatefulPartitionedCallStatefulPartitionedCall:=9
7
_user_specified_namegru/gru_cell/recurrent_kernel:3/
-
_user_specified_namegru/gru_cell/kernel:1-
+
_user_specified_namegru/gru_cell/bias:^ Z
4
_output_shapes"
 :������������������
"
_user_specified_name
inputs_0
�9
�
while_body_3185
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
strided_slice_1_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0=
+gru_cell_readvariableop_gru_gru_cell_bias_0:`F
4gru_cell_matmul_readvariableop_gru_gru_cell_kernel_0:`R
@gru_cell_matmul_1_readvariableop_gru_gru_cell_recurrent_kernel_0: `
identity

identity_1

identity_2

identity_3

identity_4
strided_slice_1O
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor;
)gru_cell_readvariableop_gru_gru_cell_bias:`D
2gru_cell_matmul_readvariableop_gru_gru_cell_kernel:`P
>gru_cell_matmul_1_readvariableop_gru_gru_cell_recurrent_kernel: `��gru_cell/MatMul/ReadVariableOp� gru_cell/MatMul_1/ReadVariableOp�gru_cell/ReadVariableOp�
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
gru_cell/ReadVariableOpReadVariableOp+gru_cell_readvariableop_gru_gru_cell_bias_0*
_output_shapes

:`*
dtype0q
gru_cell/unstackUnpackgru_cell/ReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
num�
gru_cell/MatMul/ReadVariableOpReadVariableOp4gru_cell_matmul_readvariableop_gru_gru_cell_kernel_0*
_output_shapes

:`*
dtype0�
gru_cell/MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0&gru_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������`�
gru_cell/BiasAddBiasAddgru_cell/MatMul:product:0gru_cell/unstack:output:0*
T0*'
_output_shapes
:���������`c
gru_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
gru_cell/splitSplit!gru_cell/split/split_dim:output:0gru_cell/BiasAdd:output:0*
T0*M
_output_shapes;
9:��������� :��������� :��������� *
	num_split�
 gru_cell/MatMul_1/ReadVariableOpReadVariableOp@gru_cell_matmul_1_readvariableop_gru_gru_cell_recurrent_kernel_0*
_output_shapes

: `*
dtype0�
gru_cell/MatMul_1MatMulplaceholder_2(gru_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������`�
gru_cell/BiasAdd_1BiasAddgru_cell/MatMul_1:product:0gru_cell/unstack:output:1*
T0*'
_output_shapes
:���������`c
gru_cell/ConstConst*
_output_shapes
:*
dtype0*!
valueB"        ����e
gru_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
gru_cell/split_1SplitVgru_cell/BiasAdd_1:output:0gru_cell/Const:output:0#gru_cell/split_1/split_dim:output:0*

Tlen0*
T0*M
_output_shapes;
9:��������� :��������� :��������� *
	num_split{
gru_cell/addAddV2gru_cell/split:output:0gru_cell/split_1:output:0*
T0*'
_output_shapes
:��������� _
gru_cell/SigmoidSigmoidgru_cell/add:z:0*
T0*'
_output_shapes
:��������� }
gru_cell/add_1AddV2gru_cell/split:output:1gru_cell/split_1:output:1*
T0*'
_output_shapes
:��������� c
gru_cell/Sigmoid_1Sigmoidgru_cell/add_1:z:0*
T0*'
_output_shapes
:��������� x
gru_cell/mulMulgru_cell/Sigmoid_1:y:0gru_cell/split_1:output:2*
T0*'
_output_shapes
:��������� t
gru_cell/add_2AddV2gru_cell/split:output:2gru_cell/mul:z:0*
T0*'
_output_shapes
:��������� [
gru_cell/TanhTanhgru_cell/add_2:z:0*
T0*'
_output_shapes
:��������� l
gru_cell/mul_1Mulgru_cell/Sigmoid:y:0placeholder_2*
T0*'
_output_shapes
:��������� S
gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?t
gru_cell/subSubgru_cell/sub/x:output:0gru_cell/Sigmoid:y:0*
T0*'
_output_shapes
:��������� l
gru_cell/mul_2Mulgru_cell/sub:z:0gru_cell/Tanh:y:0*
T0*'
_output_shapes
:��������� q
gru_cell/add_3AddV2gru_cell/mul_1:z:0gru_cell/mul_2:z:0*
T0*'
_output_shapes
:��������� l
*TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : �
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_13TensorArrayV2Write/TensorListSetItem/index:output:0gru_cell/add_3:z:0*
_output_shapes
: *
element_dtype0:���G
add/yConst*
_output_shapes
: *
dtype0*
value	B :J
addAddV2placeholderadd/y:output:0*
T0*
_output_shapes
: I
add_1/yConst*
_output_shapes
: *
dtype0*
value	B :U
add_1AddV2while_loop_counteradd_1/y:output:0*
T0*
_output_shapes
: G
IdentityIdentity	add_1:z:0^NoOp*
T0*
_output_shapes
: X

Identity_1Identitywhile_maximum_iterations^NoOp*
T0*
_output_shapes
: G

Identity_2Identityadd:z:0^NoOp*
T0*
_output_shapes
: t

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^NoOp*
T0*
_output_shapes
: c

Identity_4Identitygru_cell/add_3:z:0^NoOp*
T0*'
_output_shapes
:��������� �
NoOpNoOp^gru_cell/MatMul/ReadVariableOp!^gru_cell/MatMul_1/ReadVariableOp^gru_cell/ReadVariableOp*
_output_shapes
 "�
>gru_cell_matmul_1_readvariableop_gru_gru_cell_recurrent_kernel@gru_cell_matmul_1_readvariableop_gru_gru_cell_recurrent_kernel_0"j
2gru_cell_matmul_readvariableop_gru_gru_cell_kernel4gru_cell_matmul_readvariableop_gru_gru_cell_kernel_0"X
)gru_cell_readvariableop_gru_gru_cell_bias+gru_cell_readvariableop_gru_gru_cell_bias_0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
identityIdentity:output:0"$
strided_slice_1strided_slice_1_0"�
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :��������� : : : : : 2@
gru_cell/MatMul/ReadVariableOpgru_cell/MatMul/ReadVariableOp2D
 gru_cell/MatMul_1/ReadVariableOp gru_cell/MatMul_1/ReadVariableOp22
gru_cell/ReadVariableOpgru_cell/ReadVariableOp:=	9
7
_user_specified_namegru/gru_cell/recurrent_kernel:3/
-
_user_specified_namegru/gru_cell/kernel:1-
+
_user_specified_namegru/gru_cell/bias:_[

_output_shapes
: 
A
_user_specified_name)'TensorArrayUnstack/TensorListFromTensor:GC

_output_shapes
: 
)
_user_specified_namestrided_slice_1:-)
'
_output_shapes
:��������� :

_output_shapes
: :

_output_shapes
: :PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter
�
N
"__inference_dot_layer_call_fn_2566
inputs_0
inputs_1
identity�
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:�����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *F
fAR?
=__inference_dot_layer_call_and_return_conditional_losses_2079f
IdentityIdentityPartitionedCall:output:0*
T0*-
_output_shapes
:�����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:����������
:����������
:VR
,
_output_shapes
:����������

"
_user_specified_name
inputs_1:V R
,
_output_shapes
:����������

"
_user_specified_name
inputs_0
�

�
model_gru_while_cond_1629 
model_gru_while_loop_counter&
"model_gru_while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3"
less_model_gru_strided_slice_16
2model_gru_while_cond_1629___redundant_placeholder06
2model_gru_while_cond_1629___redundant_placeholder16
2model_gru_while_cond_1629___redundant_placeholder26
2model_gru_while_cond_1629___redundant_placeholder36
2model_gru_while_cond_1629___redundant_placeholder4
identity
Z
LessLessplaceholderless_model_gru_strided_slice_1*
T0*
_output_shapes
: ?
IdentityIdentityLess:z:0*
T0
*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*W
_input_shapesF
D: : : : :��������� :��������� : ::::::

_output_shapes
::

_output_shapes
::QM

_output_shapes
: 
3
_user_specified_namemodel/gru/strided_slice_1:-)
'
_output_shapes
:��������� :-)
'
_output_shapes
:��������� :

_output_shapes
: :

_output_shapes
: :ZV

_output_shapes
: 
<
_user_specified_name$"model/gru/while/maximum_iterations:T P

_output_shapes
: 
6
_user_specified_namemodel/gru/while/loop_counter
�
�
while_cond_2874
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
less_strided_slice_1,
(while_cond_2874___redundant_placeholder0,
(while_cond_2874___redundant_placeholder1,
(while_cond_2874___redundant_placeholder2,
(while_cond_2874___redundant_placeholder3
identity
P
LessLessplaceholderless_strided_slice_1*
T0*
_output_shapes
: ?
IdentityIdentityLess:z:0*
T0
*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :��������� : :::::

_output_shapes
::GC

_output_shapes
: 
)
_user_specified_namestrided_slice_1:-)
'
_output_shapes
:��������� :

_output_shapes
: :

_output_shapes
: :PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter
�
�
while_cond_1817
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
less_strided_slice_1,
(while_cond_1817___redundant_placeholder0,
(while_cond_1817___redundant_placeholder1,
(while_cond_1817___redundant_placeholder2,
(while_cond_1817___redundant_placeholder3
identity
P
LessLessplaceholderless_strided_slice_1*
T0*
_output_shapes
: ?
IdentityIdentityLess:z:0*
T0
*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :��������� : :::::

_output_shapes
::GC

_output_shapes
: 
)
_user_specified_namestrided_slice_1:-)
'
_output_shapes
:��������� :

_output_shapes
: :

_output_shapes
: :PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter
�

�
"__inference_signature_wrapper_2544
input_prior#
gru_gru_cell_bias:`%
gru_gru_cell_kernel:`/
gru_gru_cell_recurrent_kernel: `
dense_kernel: 

dense_bias:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_priorgru_gru_cell_biasgru_gru_cell_kernelgru_gru_cell_recurrent_kerneldense_kernel
dense_bias*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8� *(
f#R!
__inference__wrapped_model_1742o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":����������
: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:*&
$
_user_specified_name
dense/bias:,(
&
_user_specified_namedense/kernel:=9
7
_user_specified_namegru/gru_cell/recurrent_kernel:3/
-
_user_specified_namegru/gru_cell/kernel:1-
+
_user_specified_namegru/gru_cell/bias:Y U
,
_output_shapes
:����������

%
_user_specified_nameinput_prior
�9
�
while_body_2720
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
strided_slice_1_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0=
+gru_cell_readvariableop_gru_gru_cell_bias_0:`F
4gru_cell_matmul_readvariableop_gru_gru_cell_kernel_0:`R
@gru_cell_matmul_1_readvariableop_gru_gru_cell_recurrent_kernel_0: `
identity

identity_1

identity_2

identity_3

identity_4
strided_slice_1O
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor;
)gru_cell_readvariableop_gru_gru_cell_bias:`D
2gru_cell_matmul_readvariableop_gru_gru_cell_kernel:`P
>gru_cell_matmul_1_readvariableop_gru_gru_cell_recurrent_kernel: `��gru_cell/MatMul/ReadVariableOp� gru_cell/MatMul_1/ReadVariableOp�gru_cell/ReadVariableOp�
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
gru_cell/ReadVariableOpReadVariableOp+gru_cell_readvariableop_gru_gru_cell_bias_0*
_output_shapes

:`*
dtype0q
gru_cell/unstackUnpackgru_cell/ReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
num�
gru_cell/MatMul/ReadVariableOpReadVariableOp4gru_cell_matmul_readvariableop_gru_gru_cell_kernel_0*
_output_shapes

:`*
dtype0�
gru_cell/MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0&gru_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������`�
gru_cell/BiasAddBiasAddgru_cell/MatMul:product:0gru_cell/unstack:output:0*
T0*'
_output_shapes
:���������`c
gru_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
gru_cell/splitSplit!gru_cell/split/split_dim:output:0gru_cell/BiasAdd:output:0*
T0*M
_output_shapes;
9:��������� :��������� :��������� *
	num_split�
 gru_cell/MatMul_1/ReadVariableOpReadVariableOp@gru_cell_matmul_1_readvariableop_gru_gru_cell_recurrent_kernel_0*
_output_shapes

: `*
dtype0�
gru_cell/MatMul_1MatMulplaceholder_2(gru_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������`�
gru_cell/BiasAdd_1BiasAddgru_cell/MatMul_1:product:0gru_cell/unstack:output:1*
T0*'
_output_shapes
:���������`c
gru_cell/ConstConst*
_output_shapes
:*
dtype0*!
valueB"        ����e
gru_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
gru_cell/split_1SplitVgru_cell/BiasAdd_1:output:0gru_cell/Const:output:0#gru_cell/split_1/split_dim:output:0*

Tlen0*
T0*M
_output_shapes;
9:��������� :��������� :��������� *
	num_split{
gru_cell/addAddV2gru_cell/split:output:0gru_cell/split_1:output:0*
T0*'
_output_shapes
:��������� _
gru_cell/SigmoidSigmoidgru_cell/add:z:0*
T0*'
_output_shapes
:��������� }
gru_cell/add_1AddV2gru_cell/split:output:1gru_cell/split_1:output:1*
T0*'
_output_shapes
:��������� c
gru_cell/Sigmoid_1Sigmoidgru_cell/add_1:z:0*
T0*'
_output_shapes
:��������� x
gru_cell/mulMulgru_cell/Sigmoid_1:y:0gru_cell/split_1:output:2*
T0*'
_output_shapes
:��������� t
gru_cell/add_2AddV2gru_cell/split:output:2gru_cell/mul:z:0*
T0*'
_output_shapes
:��������� [
gru_cell/TanhTanhgru_cell/add_2:z:0*
T0*'
_output_shapes
:��������� l
gru_cell/mul_1Mulgru_cell/Sigmoid:y:0placeholder_2*
T0*'
_output_shapes
:��������� S
gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?t
gru_cell/subSubgru_cell/sub/x:output:0gru_cell/Sigmoid:y:0*
T0*'
_output_shapes
:��������� l
gru_cell/mul_2Mulgru_cell/sub:z:0gru_cell/Tanh:y:0*
T0*'
_output_shapes
:��������� q
gru_cell/add_3AddV2gru_cell/mul_1:z:0gru_cell/mul_2:z:0*
T0*'
_output_shapes
:��������� l
*TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : �
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_13TensorArrayV2Write/TensorListSetItem/index:output:0gru_cell/add_3:z:0*
_output_shapes
: *
element_dtype0:���G
add/yConst*
_output_shapes
: *
dtype0*
value	B :J
addAddV2placeholderadd/y:output:0*
T0*
_output_shapes
: I
add_1/yConst*
_output_shapes
: *
dtype0*
value	B :U
add_1AddV2while_loop_counteradd_1/y:output:0*
T0*
_output_shapes
: G
IdentityIdentity	add_1:z:0^NoOp*
T0*
_output_shapes
: X

Identity_1Identitywhile_maximum_iterations^NoOp*
T0*
_output_shapes
: G

Identity_2Identityadd:z:0^NoOp*
T0*
_output_shapes
: t

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^NoOp*
T0*
_output_shapes
: c

Identity_4Identitygru_cell/add_3:z:0^NoOp*
T0*'
_output_shapes
:��������� �
NoOpNoOp^gru_cell/MatMul/ReadVariableOp!^gru_cell/MatMul_1/ReadVariableOp^gru_cell/ReadVariableOp*
_output_shapes
 "�
>gru_cell_matmul_1_readvariableop_gru_gru_cell_recurrent_kernel@gru_cell_matmul_1_readvariableop_gru_gru_cell_recurrent_kernel_0"j
2gru_cell_matmul_readvariableop_gru_gru_cell_kernel4gru_cell_matmul_readvariableop_gru_gru_cell_kernel_0"X
)gru_cell_readvariableop_gru_gru_cell_bias+gru_cell_readvariableop_gru_gru_cell_bias_0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
identityIdentity:output:0"$
strided_slice_1strided_slice_1_0"�
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :��������� : : : : : 2@
gru_cell/MatMul/ReadVariableOpgru_cell/MatMul/ReadVariableOp2D
 gru_cell/MatMul_1/ReadVariableOp gru_cell/MatMul_1/ReadVariableOp22
gru_cell/ReadVariableOpgru_cell/ReadVariableOp:=	9
7
_user_specified_namegru/gru_cell/recurrent_kernel:3/
-
_user_specified_namegru/gru_cell/kernel:1-
+
_user_specified_namegru/gru_cell/bias:_[

_output_shapes
: 
A
_user_specified_name)'TensorArrayUnstack/TensorListFromTensor:GC

_output_shapes
: 
)
_user_specified_namestrided_slice_1:-)
'
_output_shapes
:��������� :

_output_shapes
: :

_output_shapes
: :PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter
�!
�
while_body_1818
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
strided_slice_1_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0.
gru_cell_gru_gru_cell_bias_0:`0
gru_cell_gru_gru_cell_kernel_0:`:
(gru_cell_gru_gru_cell_recurrent_kernel_0: `
identity

identity_1

identity_2

identity_3

identity_4
strided_slice_1O
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor,
gru_cell_gru_gru_cell_bias:`.
gru_cell_gru_gru_cell_kernel:`8
&gru_cell_gru_gru_cell_recurrent_kernel: `�� gru_cell/StatefulPartitionedCall�
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
 gru_cell/StatefulPartitionedCallStatefulPartitionedCall*TensorArrayV2Read/TensorListGetItem:item:0placeholder_2gru_cell_gru_gru_cell_bias_0gru_cell_gru_gru_cell_kernel_0(gru_cell_gru_gru_cell_recurrent_kernel_0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:��������� :��������� *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_gru_cell_layer_call_and_return_conditional_losses_1807l
*TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : �
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_13TensorArrayV2Write/TensorListSetItem/index:output:0)gru_cell/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:���G
add/yConst*
_output_shapes
: *
dtype0*
value	B :J
addAddV2placeholderadd/y:output:0*
T0*
_output_shapes
: I
add_1/yConst*
_output_shapes
: *
dtype0*
value	B :U
add_1AddV2while_loop_counteradd_1/y:output:0*
T0*
_output_shapes
: G
IdentityIdentity	add_1:z:0^NoOp*
T0*
_output_shapes
: X

Identity_1Identitywhile_maximum_iterations^NoOp*
T0*
_output_shapes
: G

Identity_2Identityadd:z:0^NoOp*
T0*
_output_shapes
: t

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^NoOp*
T0*
_output_shapes
: z

Identity_4Identity)gru_cell/StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:��������� E
NoOpNoOp!^gru_cell/StatefulPartitionedCall*
_output_shapes
 ":
gru_cell_gru_gru_cell_biasgru_cell_gru_gru_cell_bias_0">
gru_cell_gru_gru_cell_kernelgru_cell_gru_gru_cell_kernel_0"R
&gru_cell_gru_gru_cell_recurrent_kernel(gru_cell_gru_gru_cell_recurrent_kernel_0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
identityIdentity:output:0"$
strided_slice_1strided_slice_1_0"�
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :��������� : : : : : 2D
 gru_cell/StatefulPartitionedCall gru_cell/StatefulPartitionedCall:=	9
7
_user_specified_namegru/gru_cell/recurrent_kernel:3/
-
_user_specified_namegru/gru_cell/kernel:1-
+
_user_specified_namegru/gru_cell/bias:_[

_output_shapes
: 
A
_user_specified_name)'TensorArrayUnstack/TensorListFromTensor:GC

_output_shapes
: 
)
_user_specified_namestrided_slice_1:-)
'
_output_shapes
:��������� :

_output_shapes
: :

_output_shapes
: :PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter
�	
�
"__inference_gru_layer_call_fn_2647

inputs#
gru_gru_cell_bias:`%
gru_gru_cell_kernel:`/
gru_gru_cell_recurrent_kernel: `
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsgru_gru_cell_biasgru_gru_cell_kernelgru_gru_cell_recurrent_kernel*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *F
fAR?
=__inference_gru_layer_call_and_return_conditional_losses_2264o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:��������� <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:����������: : : 22
StatefulPartitionedCallStatefulPartitionedCall:=9
7
_user_specified_namegru/gru_cell/recurrent_kernel:3/
-
_user_specified_namegru/gru_cell/kernel:1-
+
_user_specified_namegru/gru_cell/bias:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
?__inference_dense_layer_call_and_return_conditional_losses_3293

inputs4
"matmul_readvariableop_dense_kernel: /
!biasadd_readvariableop_dense_bias:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpx
MatMul/ReadVariableOpReadVariableOp"matmul_readvariableop_dense_kernel*
_output_shapes

: *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������t
BiasAdd/ReadVariableOpReadVariableOp!biasadd_readvariableop_dense_bias*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:���������`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:���������S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:��������� : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:*&
$
_user_specified_name
dense/bias:,(
&
_user_specified_namedense/kernel:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�!
�
?__inference_model_layer_call_and_return_conditional_losses_2284
input_prior'
gru_gru_gru_cell_bias:`)
gru_gru_gru_cell_kernel:`3
!gru_gru_gru_cell_recurrent_kernel: `$
dense_dense_kernel: 
dense_dense_bias:
identity��dense/StatefulPartitionedCall�gru/StatefulPartitionedCall�
masking_prior/PartitionedCallPartitionedCallinput_prior*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_masking_prior_layer_call_and_return_conditional_losses_2069�
dot/PartitionedCallPartitionedCall&masking_prior/PartitionedCall:output:0&masking_prior/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:�����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *F
fAR?
=__inference_dot_layer_call_and_return_conditional_losses_2079�
activation/PartitionedCallPartitionedCalldot/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:�����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_activation_layer_call_and_return_conditional_losses_2085�
dot_1/PartitionedCallPartitionedCall#activation/PartitionedCall:output:0&masking_prior/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *H
fCRA
?__inference_dot_1_layer_call_and_return_conditional_losses_2093�
multiply/PartitionedCallPartitionedCalldot_1/PartitionedCall:output:0&masking_prior/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_multiply_layer_call_and_return_conditional_losses_2100�
concatenate/PartitionedCallPartitionedCall&masking_prior/PartitionedCall:output:0!multiply/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_concatenate_layer_call_and_return_conditional_losses_2108�
gru/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0gru_gru_gru_cell_biasgru_gru_gru_cell_kernel!gru_gru_gru_cell_recurrent_kernel*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *F
fAR?
=__inference_gru_layer_call_and_return_conditional_losses_2264�
dense/StatefulPartitionedCallStatefulPartitionedCall$gru/StatefulPartitionedCall:output:0dense_dense_kerneldense_dense_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_2279u
IdentityIdentity&dense/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^dense/StatefulPartitionedCall^gru/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":����������
: : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2:
gru/StatefulPartitionedCallgru/StatefulPartitionedCall:*&
$
_user_specified_name
dense/bias:,(
&
_user_specified_namedense/kernel:=9
7
_user_specified_namegru/gru_cell/recurrent_kernel:3/
-
_user_specified_namegru/gru_cell/kernel:1-
+
_user_specified_namegru/gru_cell/bias:Y U
,
_output_shapes
:����������

%
_user_specified_nameinput_prior
�
�
$__inference_model_layer_call_fn_2466
input_prior#
gru_gru_cell_bias:`%
gru_gru_cell_kernel:`/
gru_gru_cell_recurrent_kernel: `
dense_kernel: 

dense_bias:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_priorgru_gru_cell_biasgru_gru_cell_kernelgru_gru_cell_recurrent_kerneldense_kernel
dense_bias*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8� *H
fCRA
?__inference_model_layer_call_and_return_conditional_losses_2284o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":����������
: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:*&
$
_user_specified_name
dense/bias:,(
&
_user_specified_namedense/kernel:=9
7
_user_specified_namegru/gru_cell/recurrent_kernel:3/
-
_user_specified_namegru/gru_cell/kernel:1-
+
_user_specified_namegru/gru_cell/bias:Y U
,
_output_shapes
:����������

%
_user_specified_nameinput_prior
�
k
?__inference_dot_1_layer_call_and_return_conditional_losses_2598
inputs_0
inputs_1
identityb
MatMulBatchMatMulV2inputs_0inputs_1*
T0*,
_output_shapes
:����������
R
ShapeShapeMatMul:output:0*
T0*
_output_shapes
::��\
IdentityIdentityMatMul:output:0*
T0*,
_output_shapes
:����������
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:�����������:����������
:VR
,
_output_shapes
:����������

"
_user_specified_name
inputs_1:W S
-
_output_shapes
:�����������
"
_user_specified_name
inputs_0
�N
�
=__inference_gru_layer_call_and_return_conditional_losses_2965
inputs_0;
)gru_cell_readvariableop_gru_gru_cell_bias:`D
2gru_cell_matmul_readvariableop_gru_gru_cell_kernel:`P
>gru_cell_matmul_1_readvariableop_gru_gru_cell_recurrent_kernel: `
identity��gru_cell/MatMul/ReadVariableOp� gru_cell/MatMul_1/ReadVariableOp�gru_cell/ReadVariableOp�whileK
ShapeShapeinputs_0*
T0*
_output_shapes
::��]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:��������� c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :������������������R
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
::��_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask�
gru_cell/ReadVariableOpReadVariableOp)gru_cell_readvariableop_gru_gru_cell_bias*
_output_shapes

:`*
dtype0q
gru_cell/unstackUnpackgru_cell/ReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
num�
gru_cell/MatMul/ReadVariableOpReadVariableOp2gru_cell_matmul_readvariableop_gru_gru_cell_kernel*
_output_shapes

:`*
dtype0�
gru_cell/MatMulMatMulstrided_slice_2:output:0&gru_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������`�
gru_cell/BiasAddBiasAddgru_cell/MatMul:product:0gru_cell/unstack:output:0*
T0*'
_output_shapes
:���������`c
gru_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
gru_cell/splitSplit!gru_cell/split/split_dim:output:0gru_cell/BiasAdd:output:0*
T0*M
_output_shapes;
9:��������� :��������� :��������� *
	num_split�
 gru_cell/MatMul_1/ReadVariableOpReadVariableOp>gru_cell_matmul_1_readvariableop_gru_gru_cell_recurrent_kernel*
_output_shapes

: `*
dtype0�
gru_cell/MatMul_1MatMulzeros:output:0(gru_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������`�
gru_cell/BiasAdd_1BiasAddgru_cell/MatMul_1:product:0gru_cell/unstack:output:1*
T0*'
_output_shapes
:���������`c
gru_cell/ConstConst*
_output_shapes
:*
dtype0*!
valueB"        ����e
gru_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
gru_cell/split_1SplitVgru_cell/BiasAdd_1:output:0gru_cell/Const:output:0#gru_cell/split_1/split_dim:output:0*

Tlen0*
T0*M
_output_shapes;
9:��������� :��������� :��������� *
	num_split{
gru_cell/addAddV2gru_cell/split:output:0gru_cell/split_1:output:0*
T0*'
_output_shapes
:��������� _
gru_cell/SigmoidSigmoidgru_cell/add:z:0*
T0*'
_output_shapes
:��������� }
gru_cell/add_1AddV2gru_cell/split:output:1gru_cell/split_1:output:1*
T0*'
_output_shapes
:��������� c
gru_cell/Sigmoid_1Sigmoidgru_cell/add_1:z:0*
T0*'
_output_shapes
:��������� x
gru_cell/mulMulgru_cell/Sigmoid_1:y:0gru_cell/split_1:output:2*
T0*'
_output_shapes
:��������� t
gru_cell/add_2AddV2gru_cell/split:output:2gru_cell/mul:z:0*
T0*'
_output_shapes
:��������� [
gru_cell/TanhTanhgru_cell/add_2:z:0*
T0*'
_output_shapes
:��������� m
gru_cell/mul_1Mulgru_cell/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:��������� S
gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?t
gru_cell/subSubgru_cell/sub/x:output:0gru_cell/Sigmoid:y:0*
T0*'
_output_shapes
:��������� l
gru_cell/mul_2Mulgru_cell/sub:z:0gru_cell/Tanh:y:0*
T0*'
_output_shapes
:��������� q
gru_cell/add_3AddV2gru_cell/mul_1:z:0gru_cell/mul_2:z:0*
T0*'
_output_shapes
:��������� n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :�
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)gru_cell_readvariableop_gru_gru_cell_bias2gru_cell_matmul_readvariableop_gru_gru_cell_kernel>gru_cell_matmul_1_readvariableop_gru_gru_cell_recurrent_kernel*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :��������� : : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_2875*
condR
while_cond_2874*8
output_shapes'
%: : : : :��������� : : : : : *
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:��������� *
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:��������� *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:��������� [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:��������� �
NoOpNoOp^gru_cell/MatMul/ReadVariableOp!^gru_cell/MatMul_1/ReadVariableOp^gru_cell/ReadVariableOp^while*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������: : : 2@
gru_cell/MatMul/ReadVariableOpgru_cell/MatMul/ReadVariableOp2D
 gru_cell/MatMul_1/ReadVariableOp gru_cell/MatMul_1/ReadVariableOp22
gru_cell/ReadVariableOpgru_cell/ReadVariableOp2
whilewhile:=9
7
_user_specified_namegru/gru_cell/recurrent_kernel:3/
-
_user_specified_namegru/gru_cell/kernel:1-
+
_user_specified_namegru/gru_cell/bias:^ Z
4
_output_shapes"
 :������������������
"
_user_specified_name
inputs_0
�6
�
=__inference_gru_layer_call_and_return_conditional_losses_1880

inputs,
gru_cell_gru_gru_cell_bias:`.
gru_cell_gru_gru_cell_kernel:`8
&gru_cell_gru_gru_cell_recurrent_kernel: `
identity�� gru_cell/StatefulPartitionedCall�whileI
ShapeShapeinputs*
T0*
_output_shapes
::��]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:��������� c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :������������������R
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
::��_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask�
 gru_cell/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0gru_cell_gru_gru_cell_biasgru_cell_gru_gru_cell_kernel&gru_cell_gru_gru_cell_recurrent_kernel*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:��������� :��������� *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_gru_cell_layer_call_and_return_conditional_losses_1807n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :�
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0gru_cell_gru_gru_cell_biasgru_cell_gru_gru_cell_kernel&gru_cell_gru_gru_cell_recurrent_kernel*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :��������� : : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_1818*
condR
while_cond_1817*8
output_shapes'
%: : : : :��������� : : : : : *
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:��������� *
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:��������� *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:��������� [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:��������� M
NoOpNoOp!^gru_cell/StatefulPartitionedCall^while*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������: : : 2D
 gru_cell/StatefulPartitionedCall gru_cell/StatefulPartitionedCall2
whilewhile:=9
7
_user_specified_namegru/gru_cell/recurrent_kernel:3/
-
_user_specified_namegru/gru_cell/kernel:1-
+
_user_specified_namegru/gru_cell/bias:\ X
4
_output_shapes"
 :������������������
 
_user_specified_nameinputs
�
i
=__inference_dot_layer_call_and_return_conditional_losses_2575
inputs_0
inputs_1
identityc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          p
	transpose	Transposeinputs_1transpose/perm:output:0*
T0*,
_output_shapes
:���������
�h
MatMulBatchMatMulV2inputs_0transpose:y:0*
T0*-
_output_shapes
:�����������R
ShapeShapeMatMul:output:0*
T0*
_output_shapes
::��]
IdentityIdentityMatMul:output:0*
T0*-
_output_shapes
:�����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:����������
:����������
:VR
,
_output_shapes
:����������

"
_user_specified_name
inputs_1:V R
,
_output_shapes
:����������

"
_user_specified_name
inputs_0
�
�
'__inference_gru_cell_layer_call_fn_3304

inputs
states_0#
gru_gru_cell_bias:`%
gru_gru_cell_kernel:`/
gru_gru_cell_recurrent_kernel: `
identity

identity_1��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0gru_gru_cell_biasgru_gru_cell_kernelgru_gru_cell_recurrent_kernel*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:��������� :��������� *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_gru_cell_layer_call_and_return_conditional_losses_1807o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:��������� q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:��������� <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:���������:��������� : : : 22
StatefulPartitionedCallStatefulPartitionedCall:=9
7
_user_specified_namegru/gru_cell/recurrent_kernel:3/
-
_user_specified_namegru/gru_cell/kernel:1-
+
_user_specified_namegru/gru_cell/bias:QM
'
_output_shapes
:��������� 
"
_user_specified_name
states_0:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�!
�
?__inference_model_layer_call_and_return_conditional_losses_2456
input_prior'
gru_gru_gru_cell_bias:`)
gru_gru_gru_cell_kernel:`3
!gru_gru_gru_cell_recurrent_kernel: `$
dense_dense_kernel: 
dense_dense_bias:
identity��dense/StatefulPartitionedCall�gru/StatefulPartitionedCall�
masking_prior/PartitionedCallPartitionedCallinput_prior*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_masking_prior_layer_call_and_return_conditional_losses_2069�
dot/PartitionedCallPartitionedCall&masking_prior/PartitionedCall:output:0&masking_prior/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:�����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *F
fAR?
=__inference_dot_layer_call_and_return_conditional_losses_2079�
activation/PartitionedCallPartitionedCalldot/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:�����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_activation_layer_call_and_return_conditional_losses_2085�
dot_1/PartitionedCallPartitionedCall#activation/PartitionedCall:output:0&masking_prior/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *H
fCRA
?__inference_dot_1_layer_call_and_return_conditional_losses_2093�
multiply/PartitionedCallPartitionedCalldot_1/PartitionedCall:output:0&masking_prior/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_multiply_layer_call_and_return_conditional_losses_2100�
concatenate/PartitionedCallPartitionedCall&masking_prior/PartitionedCall:output:0!multiply/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_concatenate_layer_call_and_return_conditional_losses_2108�
gru/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0gru_gru_gru_cell_biasgru_gru_gru_cell_kernel!gru_gru_gru_cell_recurrent_kernel*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *F
fAR?
=__inference_gru_layer_call_and_return_conditional_losses_2447�
dense/StatefulPartitionedCallStatefulPartitionedCall$gru/StatefulPartitionedCall:output:0dense_dense_kerneldense_dense_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_2279u
IdentityIdentity&dense/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^dense/StatefulPartitionedCall^gru/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":����������
: : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2:
gru/StatefulPartitionedCallgru/StatefulPartitionedCall:*&
$
_user_specified_name
dense/bias:,(
&
_user_specified_namedense/kernel:=9
7
_user_specified_namegru/gru_cell/recurrent_kernel:3/
-
_user_specified_namegru/gru_cell/kernel:1-
+
_user_specified_namegru/gru_cell/bias:Y U
,
_output_shapes
:����������

%
_user_specified_nameinput_prior
�
c
G__inference_masking_prior_layer_call_and_return_conditional_losses_2069

inputs
identityO

NotEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ��h
NotEqualNotEqualinputsNotEqual/y:output:0*
T0*,
_output_shapes
:����������
`
Any/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
���������w
AnyAnyNotEqual:z:0Any/reduction_indices:output:0*,
_output_shapes
:����������*
	keep_dims(`
CastCastAny:output:0*

DstT0*

SrcT0
*,
_output_shapes
:����������S
mulMulinputsCast:y:0*
T0*,
_output_shapes
:����������
s
SqueezeSqueezeAny:output:0*
T0
*(
_output_shapes
:����������*
squeeze_dims

���������T
IdentityIdentitymul:z:0*
T0*,
_output_shapes
:����������
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������
:T P
,
_output_shapes
:����������

 
_user_specified_nameinputs
�
`
D__inference_activation_layer_call_and_return_conditional_losses_2085

inputs
identityR
SoftmaxSoftmaxinputs*
T0*-
_output_shapes
:�����������_
IdentityIdentitySoftmax:softmax:0*
T0*-
_output_shapes
:�����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:�����������:U Q
-
_output_shapes
:�����������
 
_user_specified_nameinputs
�N
�
=__inference_gru_layer_call_and_return_conditional_losses_3120

inputs;
)gru_cell_readvariableop_gru_gru_cell_bias:`D
2gru_cell_matmul_readvariableop_gru_gru_cell_kernel:`P
>gru_cell_matmul_1_readvariableop_gru_gru_cell_recurrent_kernel: `
identity��gru_cell/MatMul/ReadVariableOp� gru_cell/MatMul_1/ReadVariableOp�gru_cell/ReadVariableOp�whileI
ShapeShapeinputs*
T0*
_output_shapes
::��]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:��������� c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          n
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:����������R
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
::��_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask�
gru_cell/ReadVariableOpReadVariableOp)gru_cell_readvariableop_gru_gru_cell_bias*
_output_shapes

:`*
dtype0q
gru_cell/unstackUnpackgru_cell/ReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
num�
gru_cell/MatMul/ReadVariableOpReadVariableOp2gru_cell_matmul_readvariableop_gru_gru_cell_kernel*
_output_shapes

:`*
dtype0�
gru_cell/MatMulMatMulstrided_slice_2:output:0&gru_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������`�
gru_cell/BiasAddBiasAddgru_cell/MatMul:product:0gru_cell/unstack:output:0*
T0*'
_output_shapes
:���������`c
gru_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
gru_cell/splitSplit!gru_cell/split/split_dim:output:0gru_cell/BiasAdd:output:0*
T0*M
_output_shapes;
9:��������� :��������� :��������� *
	num_split�
 gru_cell/MatMul_1/ReadVariableOpReadVariableOp>gru_cell_matmul_1_readvariableop_gru_gru_cell_recurrent_kernel*
_output_shapes

: `*
dtype0�
gru_cell/MatMul_1MatMulzeros:output:0(gru_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������`�
gru_cell/BiasAdd_1BiasAddgru_cell/MatMul_1:product:0gru_cell/unstack:output:1*
T0*'
_output_shapes
:���������`c
gru_cell/ConstConst*
_output_shapes
:*
dtype0*!
valueB"        ����e
gru_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
gru_cell/split_1SplitVgru_cell/BiasAdd_1:output:0gru_cell/Const:output:0#gru_cell/split_1/split_dim:output:0*

Tlen0*
T0*M
_output_shapes;
9:��������� :��������� :��������� *
	num_split{
gru_cell/addAddV2gru_cell/split:output:0gru_cell/split_1:output:0*
T0*'
_output_shapes
:��������� _
gru_cell/SigmoidSigmoidgru_cell/add:z:0*
T0*'
_output_shapes
:��������� }
gru_cell/add_1AddV2gru_cell/split:output:1gru_cell/split_1:output:1*
T0*'
_output_shapes
:��������� c
gru_cell/Sigmoid_1Sigmoidgru_cell/add_1:z:0*
T0*'
_output_shapes
:��������� x
gru_cell/mulMulgru_cell/Sigmoid_1:y:0gru_cell/split_1:output:2*
T0*'
_output_shapes
:��������� t
gru_cell/add_2AddV2gru_cell/split:output:2gru_cell/mul:z:0*
T0*'
_output_shapes
:��������� [
gru_cell/TanhTanhgru_cell/add_2:z:0*
T0*'
_output_shapes
:��������� m
gru_cell/mul_1Mulgru_cell/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:��������� S
gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?t
gru_cell/subSubgru_cell/sub/x:output:0gru_cell/Sigmoid:y:0*
T0*'
_output_shapes
:��������� l
gru_cell/mul_2Mulgru_cell/sub:z:0gru_cell/Tanh:y:0*
T0*'
_output_shapes
:��������� q
gru_cell/add_3AddV2gru_cell/mul_1:z:0gru_cell/mul_2:z:0*
T0*'
_output_shapes
:��������� n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :�
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)gru_cell_readvariableop_gru_gru_cell_bias2gru_cell_matmul_readvariableop_gru_gru_cell_kernel>gru_cell_matmul_1_readvariableop_gru_gru_cell_recurrent_kernel*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :��������� : : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_3030*
condR
while_cond_3029*8
output_shapes'
%: : : : :��������� : : : : : *
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:��������� *
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:��������� *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:��������� [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:��������� �
NoOpNoOp^gru_cell/MatMul/ReadVariableOp!^gru_cell/MatMul_1/ReadVariableOp^gru_cell/ReadVariableOp^while*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:����������: : : 2@
gru_cell/MatMul/ReadVariableOpgru_cell/MatMul/ReadVariableOp2D
 gru_cell/MatMul_1/ReadVariableOp gru_cell/MatMul_1/ReadVariableOp22
gru_cell/ReadVariableOpgru_cell/ReadVariableOp2
whilewhile:=9
7
_user_specified_namegru/gru_cell/recurrent_kernel:3/
-
_user_specified_namegru/gru_cell/kernel:1-
+
_user_specified_namegru/gru_cell/bias:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
$__inference_dense_layer_call_fn_3282

inputs
dense_kernel: 

dense_bias:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsdense_kernel
dense_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_2279o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:��������� : : 22
StatefulPartitionedCallStatefulPartitionedCall:*&
$
_user_specified_name
dense/bias:,(
&
_user_specified_namedense/kernel:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�9
�
while_body_2357
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
strided_slice_1_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0=
+gru_cell_readvariableop_gru_gru_cell_bias_0:`F
4gru_cell_matmul_readvariableop_gru_gru_cell_kernel_0:`R
@gru_cell_matmul_1_readvariableop_gru_gru_cell_recurrent_kernel_0: `
identity

identity_1

identity_2

identity_3

identity_4
strided_slice_1O
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor;
)gru_cell_readvariableop_gru_gru_cell_bias:`D
2gru_cell_matmul_readvariableop_gru_gru_cell_kernel:`P
>gru_cell_matmul_1_readvariableop_gru_gru_cell_recurrent_kernel: `��gru_cell/MatMul/ReadVariableOp� gru_cell/MatMul_1/ReadVariableOp�gru_cell/ReadVariableOp�
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
gru_cell/ReadVariableOpReadVariableOp+gru_cell_readvariableop_gru_gru_cell_bias_0*
_output_shapes

:`*
dtype0q
gru_cell/unstackUnpackgru_cell/ReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
num�
gru_cell/MatMul/ReadVariableOpReadVariableOp4gru_cell_matmul_readvariableop_gru_gru_cell_kernel_0*
_output_shapes

:`*
dtype0�
gru_cell/MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0&gru_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������`�
gru_cell/BiasAddBiasAddgru_cell/MatMul:product:0gru_cell/unstack:output:0*
T0*'
_output_shapes
:���������`c
gru_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
gru_cell/splitSplit!gru_cell/split/split_dim:output:0gru_cell/BiasAdd:output:0*
T0*M
_output_shapes;
9:��������� :��������� :��������� *
	num_split�
 gru_cell/MatMul_1/ReadVariableOpReadVariableOp@gru_cell_matmul_1_readvariableop_gru_gru_cell_recurrent_kernel_0*
_output_shapes

: `*
dtype0�
gru_cell/MatMul_1MatMulplaceholder_2(gru_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������`�
gru_cell/BiasAdd_1BiasAddgru_cell/MatMul_1:product:0gru_cell/unstack:output:1*
T0*'
_output_shapes
:���������`c
gru_cell/ConstConst*
_output_shapes
:*
dtype0*!
valueB"        ����e
gru_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
gru_cell/split_1SplitVgru_cell/BiasAdd_1:output:0gru_cell/Const:output:0#gru_cell/split_1/split_dim:output:0*

Tlen0*
T0*M
_output_shapes;
9:��������� :��������� :��������� *
	num_split{
gru_cell/addAddV2gru_cell/split:output:0gru_cell/split_1:output:0*
T0*'
_output_shapes
:��������� _
gru_cell/SigmoidSigmoidgru_cell/add:z:0*
T0*'
_output_shapes
:��������� }
gru_cell/add_1AddV2gru_cell/split:output:1gru_cell/split_1:output:1*
T0*'
_output_shapes
:��������� c
gru_cell/Sigmoid_1Sigmoidgru_cell/add_1:z:0*
T0*'
_output_shapes
:��������� x
gru_cell/mulMulgru_cell/Sigmoid_1:y:0gru_cell/split_1:output:2*
T0*'
_output_shapes
:��������� t
gru_cell/add_2AddV2gru_cell/split:output:2gru_cell/mul:z:0*
T0*'
_output_shapes
:��������� [
gru_cell/TanhTanhgru_cell/add_2:z:0*
T0*'
_output_shapes
:��������� l
gru_cell/mul_1Mulgru_cell/Sigmoid:y:0placeholder_2*
T0*'
_output_shapes
:��������� S
gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?t
gru_cell/subSubgru_cell/sub/x:output:0gru_cell/Sigmoid:y:0*
T0*'
_output_shapes
:��������� l
gru_cell/mul_2Mulgru_cell/sub:z:0gru_cell/Tanh:y:0*
T0*'
_output_shapes
:��������� q
gru_cell/add_3AddV2gru_cell/mul_1:z:0gru_cell/mul_2:z:0*
T0*'
_output_shapes
:��������� l
*TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : �
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_13TensorArrayV2Write/TensorListSetItem/index:output:0gru_cell/add_3:z:0*
_output_shapes
: *
element_dtype0:���G
add/yConst*
_output_shapes
: *
dtype0*
value	B :J
addAddV2placeholderadd/y:output:0*
T0*
_output_shapes
: I
add_1/yConst*
_output_shapes
: *
dtype0*
value	B :U
add_1AddV2while_loop_counteradd_1/y:output:0*
T0*
_output_shapes
: G
IdentityIdentity	add_1:z:0^NoOp*
T0*
_output_shapes
: X

Identity_1Identitywhile_maximum_iterations^NoOp*
T0*
_output_shapes
: G

Identity_2Identityadd:z:0^NoOp*
T0*
_output_shapes
: t

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^NoOp*
T0*
_output_shapes
: c

Identity_4Identitygru_cell/add_3:z:0^NoOp*
T0*'
_output_shapes
:��������� �
NoOpNoOp^gru_cell/MatMul/ReadVariableOp!^gru_cell/MatMul_1/ReadVariableOp^gru_cell/ReadVariableOp*
_output_shapes
 "�
>gru_cell_matmul_1_readvariableop_gru_gru_cell_recurrent_kernel@gru_cell_matmul_1_readvariableop_gru_gru_cell_recurrent_kernel_0"j
2gru_cell_matmul_readvariableop_gru_gru_cell_kernel4gru_cell_matmul_readvariableop_gru_gru_cell_kernel_0"X
)gru_cell_readvariableop_gru_gru_cell_bias+gru_cell_readvariableop_gru_gru_cell_bias_0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
identityIdentity:output:0"$
strided_slice_1strided_slice_1_0"�
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :��������� : : : : : 2@
gru_cell/MatMul/ReadVariableOpgru_cell/MatMul/ReadVariableOp2D
 gru_cell/MatMul_1/ReadVariableOp gru_cell/MatMul_1/ReadVariableOp22
gru_cell/ReadVariableOpgru_cell/ReadVariableOp:=	9
7
_user_specified_namegru/gru_cell/recurrent_kernel:3/
-
_user_specified_namegru/gru_cell/kernel:1-
+
_user_specified_namegru/gru_cell/bias:_[

_output_shapes
: 
A
_user_specified_name)'TensorArrayUnstack/TensorListFromTensor:GC

_output_shapes
: 
)
_user_specified_namestrided_slice_1:-)
'
_output_shapes
:��������� :

_output_shapes
: :

_output_shapes
: :PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter
�N
�
=__inference_gru_layer_call_and_return_conditional_losses_2264

inputs;
)gru_cell_readvariableop_gru_gru_cell_bias:`D
2gru_cell_matmul_readvariableop_gru_gru_cell_kernel:`P
>gru_cell_matmul_1_readvariableop_gru_gru_cell_recurrent_kernel: `
identity��gru_cell/MatMul/ReadVariableOp� gru_cell/MatMul_1/ReadVariableOp�gru_cell/ReadVariableOp�whileI
ShapeShapeinputs*
T0*
_output_shapes
::��]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:��������� c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          n
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:����������R
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
::��_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask�
gru_cell/ReadVariableOpReadVariableOp)gru_cell_readvariableop_gru_gru_cell_bias*
_output_shapes

:`*
dtype0q
gru_cell/unstackUnpackgru_cell/ReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
num�
gru_cell/MatMul/ReadVariableOpReadVariableOp2gru_cell_matmul_readvariableop_gru_gru_cell_kernel*
_output_shapes

:`*
dtype0�
gru_cell/MatMulMatMulstrided_slice_2:output:0&gru_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������`�
gru_cell/BiasAddBiasAddgru_cell/MatMul:product:0gru_cell/unstack:output:0*
T0*'
_output_shapes
:���������`c
gru_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
gru_cell/splitSplit!gru_cell/split/split_dim:output:0gru_cell/BiasAdd:output:0*
T0*M
_output_shapes;
9:��������� :��������� :��������� *
	num_split�
 gru_cell/MatMul_1/ReadVariableOpReadVariableOp>gru_cell_matmul_1_readvariableop_gru_gru_cell_recurrent_kernel*
_output_shapes

: `*
dtype0�
gru_cell/MatMul_1MatMulzeros:output:0(gru_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������`�
gru_cell/BiasAdd_1BiasAddgru_cell/MatMul_1:product:0gru_cell/unstack:output:1*
T0*'
_output_shapes
:���������`c
gru_cell/ConstConst*
_output_shapes
:*
dtype0*!
valueB"        ����e
gru_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
gru_cell/split_1SplitVgru_cell/BiasAdd_1:output:0gru_cell/Const:output:0#gru_cell/split_1/split_dim:output:0*

Tlen0*
T0*M
_output_shapes;
9:��������� :��������� :��������� *
	num_split{
gru_cell/addAddV2gru_cell/split:output:0gru_cell/split_1:output:0*
T0*'
_output_shapes
:��������� _
gru_cell/SigmoidSigmoidgru_cell/add:z:0*
T0*'
_output_shapes
:��������� }
gru_cell/add_1AddV2gru_cell/split:output:1gru_cell/split_1:output:1*
T0*'
_output_shapes
:��������� c
gru_cell/Sigmoid_1Sigmoidgru_cell/add_1:z:0*
T0*'
_output_shapes
:��������� x
gru_cell/mulMulgru_cell/Sigmoid_1:y:0gru_cell/split_1:output:2*
T0*'
_output_shapes
:��������� t
gru_cell/add_2AddV2gru_cell/split:output:2gru_cell/mul:z:0*
T0*'
_output_shapes
:��������� [
gru_cell/TanhTanhgru_cell/add_2:z:0*
T0*'
_output_shapes
:��������� m
gru_cell/mul_1Mulgru_cell/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:��������� S
gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?t
gru_cell/subSubgru_cell/sub/x:output:0gru_cell/Sigmoid:y:0*
T0*'
_output_shapes
:��������� l
gru_cell/mul_2Mulgru_cell/sub:z:0gru_cell/Tanh:y:0*
T0*'
_output_shapes
:��������� q
gru_cell/add_3AddV2gru_cell/mul_1:z:0gru_cell/mul_2:z:0*
T0*'
_output_shapes
:��������� n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :�
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)gru_cell_readvariableop_gru_gru_cell_bias2gru_cell_matmul_readvariableop_gru_gru_cell_kernel>gru_cell_matmul_1_readvariableop_gru_gru_cell_recurrent_kernel*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :��������� : : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_2174*
condR
while_cond_2173*8
output_shapes'
%: : : : :��������� : : : : : *
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:��������� *
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:��������� *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:��������� [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:��������� �
NoOpNoOp^gru_cell/MatMul/ReadVariableOp!^gru_cell/MatMul_1/ReadVariableOp^gru_cell/ReadVariableOp^while*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:����������: : : 2@
gru_cell/MatMul/ReadVariableOpgru_cell/MatMul/ReadVariableOp2D
 gru_cell/MatMul_1/ReadVariableOp gru_cell/MatMul_1/ReadVariableOp22
gru_cell/ReadVariableOpgru_cell/ReadVariableOp2
whilewhile:=9
7
_user_specified_namegru/gru_cell/recurrent_kernel:3/
-
_user_specified_namegru/gru_cell/kernel:1-
+
_user_specified_namegru/gru_cell/bias:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�N
�
=__inference_gru_layer_call_and_return_conditional_losses_3275

inputs;
)gru_cell_readvariableop_gru_gru_cell_bias:`D
2gru_cell_matmul_readvariableop_gru_gru_cell_kernel:`P
>gru_cell_matmul_1_readvariableop_gru_gru_cell_recurrent_kernel: `
identity��gru_cell/MatMul/ReadVariableOp� gru_cell/MatMul_1/ReadVariableOp�gru_cell/ReadVariableOp�whileI
ShapeShapeinputs*
T0*
_output_shapes
::��]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:��������� c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          n
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:����������R
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
::��_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask�
gru_cell/ReadVariableOpReadVariableOp)gru_cell_readvariableop_gru_gru_cell_bias*
_output_shapes

:`*
dtype0q
gru_cell/unstackUnpackgru_cell/ReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
num�
gru_cell/MatMul/ReadVariableOpReadVariableOp2gru_cell_matmul_readvariableop_gru_gru_cell_kernel*
_output_shapes

:`*
dtype0�
gru_cell/MatMulMatMulstrided_slice_2:output:0&gru_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������`�
gru_cell/BiasAddBiasAddgru_cell/MatMul:product:0gru_cell/unstack:output:0*
T0*'
_output_shapes
:���������`c
gru_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
gru_cell/splitSplit!gru_cell/split/split_dim:output:0gru_cell/BiasAdd:output:0*
T0*M
_output_shapes;
9:��������� :��������� :��������� *
	num_split�
 gru_cell/MatMul_1/ReadVariableOpReadVariableOp>gru_cell_matmul_1_readvariableop_gru_gru_cell_recurrent_kernel*
_output_shapes

: `*
dtype0�
gru_cell/MatMul_1MatMulzeros:output:0(gru_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������`�
gru_cell/BiasAdd_1BiasAddgru_cell/MatMul_1:product:0gru_cell/unstack:output:1*
T0*'
_output_shapes
:���������`c
gru_cell/ConstConst*
_output_shapes
:*
dtype0*!
valueB"        ����e
gru_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
gru_cell/split_1SplitVgru_cell/BiasAdd_1:output:0gru_cell/Const:output:0#gru_cell/split_1/split_dim:output:0*

Tlen0*
T0*M
_output_shapes;
9:��������� :��������� :��������� *
	num_split{
gru_cell/addAddV2gru_cell/split:output:0gru_cell/split_1:output:0*
T0*'
_output_shapes
:��������� _
gru_cell/SigmoidSigmoidgru_cell/add:z:0*
T0*'
_output_shapes
:��������� }
gru_cell/add_1AddV2gru_cell/split:output:1gru_cell/split_1:output:1*
T0*'
_output_shapes
:��������� c
gru_cell/Sigmoid_1Sigmoidgru_cell/add_1:z:0*
T0*'
_output_shapes
:��������� x
gru_cell/mulMulgru_cell/Sigmoid_1:y:0gru_cell/split_1:output:2*
T0*'
_output_shapes
:��������� t
gru_cell/add_2AddV2gru_cell/split:output:2gru_cell/mul:z:0*
T0*'
_output_shapes
:��������� [
gru_cell/TanhTanhgru_cell/add_2:z:0*
T0*'
_output_shapes
:��������� m
gru_cell/mul_1Mulgru_cell/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:��������� S
gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?t
gru_cell/subSubgru_cell/sub/x:output:0gru_cell/Sigmoid:y:0*
T0*'
_output_shapes
:��������� l
gru_cell/mul_2Mulgru_cell/sub:z:0gru_cell/Tanh:y:0*
T0*'
_output_shapes
:��������� q
gru_cell/add_3AddV2gru_cell/mul_1:z:0gru_cell/mul_2:z:0*
T0*'
_output_shapes
:��������� n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :�
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)gru_cell_readvariableop_gru_gru_cell_bias2gru_cell_matmul_readvariableop_gru_gru_cell_kernel>gru_cell_matmul_1_readvariableop_gru_gru_cell_recurrent_kernel*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :��������� : : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_3185*
condR
while_cond_3184*8
output_shapes'
%: : : : :��������� : : : : : *
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:��������� *
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:��������� *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:��������� [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:��������� �
NoOpNoOp^gru_cell/MatMul/ReadVariableOp!^gru_cell/MatMul_1/ReadVariableOp^gru_cell/ReadVariableOp^while*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:����������: : : 2@
gru_cell/MatMul/ReadVariableOpgru_cell/MatMul/ReadVariableOp2D
 gru_cell/MatMul_1/ReadVariableOp gru_cell/MatMul_1/ReadVariableOp22
gru_cell/ReadVariableOpgru_cell/ReadVariableOp2
whilewhile:=9
7
_user_specified_namegru/gru_cell/recurrent_kernel:3/
-
_user_specified_namegru/gru_cell/kernel:1-
+
_user_specified_namegru/gru_cell/bias:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�
q
E__inference_concatenate_layer_call_and_return_conditional_losses_2623
inputs_0
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :|
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*,
_output_shapes
:����������\
IdentityIdentityconcat:output:0*
T0*,
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:����������
:����������
:VR
,
_output_shapes
:����������

"
_user_specified_name
inputs_1:V R
,
_output_shapes
:����������

"
_user_specified_name
inputs_0
�~
�
 __inference__traced_restore_3658
file_prefix/
assignvariableop_dense_kernel: +
assignvariableop_1_dense_bias:8
&assignvariableop_2_gru_gru_cell_kernel:`B
0assignvariableop_3_gru_gru_cell_recurrent_kernel: `6
$assignvariableop_4_gru_gru_cell_bias:`/
%assignvariableop_5_training_adam_iter:	 1
'assignvariableop_6_training_adam_beta_1: 1
'assignvariableop_7_training_adam_beta_2: 0
&assignvariableop_8_training_adam_decay: 8
.assignvariableop_9_training_adam_learning_rate: #
assignvariableop_10_total: #
assignvariableop_11_count: 1
"assignvariableop_12_true_positives:	�1
"assignvariableop_13_true_negatives:	�2
#assignvariableop_14_false_positives:	�2
#assignvariableop_15_false_negatives:	�B
0assignvariableop_16_training_adam_dense_kernel_m: <
.assignvariableop_17_training_adam_dense_bias_m:I
7assignvariableop_18_training_adam_gru_gru_cell_kernel_m:`S
Aassignvariableop_19_training_adam_gru_gru_cell_recurrent_kernel_m: `G
5assignvariableop_20_training_adam_gru_gru_cell_bias_m:`B
0assignvariableop_21_training_adam_dense_kernel_v: <
.assignvariableop_22_training_adam_dense_bias_v:I
7assignvariableop_23_training_adam_gru_gru_cell_kernel_v:`S
Aassignvariableop_24_training_adam_gru_gru_cell_recurrent_kernel_v: `G
5assignvariableop_25_training_adam_gru_gru_cell_bias_v:`
identity_27��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_3�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/1/true_positives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/1/true_negatives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/1/false_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/1/false_negatives/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*I
value@B>B B B B B B B B B B B B B B B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapesn
l:::::::::::::::::::::::::::*)
dtypes
2	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOpassignvariableop_dense_kernelIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_biasIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOp&assignvariableop_2_gru_gru_cell_kernelIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOp0assignvariableop_3_gru_gru_cell_recurrent_kernelIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOp$assignvariableop_4_gru_gru_cell_biasIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOp_5AssignVariableOp%assignvariableop_5_training_adam_iterIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOp'assignvariableop_6_training_adam_beta_1Identity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOp'assignvariableop_7_training_adam_beta_2Identity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOp&assignvariableop_8_training_adam_decayIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOp.assignvariableop_9_training_adam_learning_rateIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOpassignvariableop_10_totalIdentity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOpassignvariableop_11_countIdentity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_12AssignVariableOp"assignvariableop_12_true_positivesIdentity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOp"assignvariableop_13_true_negativesIdentity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOp#assignvariableop_14_false_positivesIdentity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOp#assignvariableop_15_false_negativesIdentity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_16AssignVariableOp0assignvariableop_16_training_adam_dense_kernel_mIdentity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_17AssignVariableOp.assignvariableop_17_training_adam_dense_bias_mIdentity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_18AssignVariableOp7assignvariableop_18_training_adam_gru_gru_cell_kernel_mIdentity_18:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_19AssignVariableOpAassignvariableop_19_training_adam_gru_gru_cell_recurrent_kernel_mIdentity_19:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_20AssignVariableOp5assignvariableop_20_training_adam_gru_gru_cell_bias_mIdentity_20:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_21AssignVariableOp0assignvariableop_21_training_adam_dense_kernel_vIdentity_21:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_22AssignVariableOp.assignvariableop_22_training_adam_dense_bias_vIdentity_22:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_23AssignVariableOp7assignvariableop_23_training_adam_gru_gru_cell_kernel_vIdentity_23:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_24AssignVariableOpAassignvariableop_24_training_adam_gru_gru_cell_recurrent_kernel_vIdentity_24:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_25AssignVariableOp5assignvariableop_25_training_adam_gru_gru_cell_bias_vIdentity_25:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 �
Identity_26Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_27IdentityIdentity_26:output:0^NoOp_1*
T0*
_output_shapes
: �
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
_output_shapes
 "#
identity_27Identity_27:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6: : : : : : : : : : : : : : : : : : : : : : : : : : : 2*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92$
AssignVariableOpAssignVariableOp:A=
;
_user_specified_name#!training/Adam/gru/gru_cell/bias/v:MI
G
_user_specified_name/-training/Adam/gru/gru_cell/recurrent_kernel/v:C?
=
_user_specified_name%#training/Adam/gru/gru_cell/kernel/v::6
4
_user_specified_nametraining/Adam/dense/bias/v:<8
6
_user_specified_nametraining/Adam/dense/kernel/v:A=
;
_user_specified_name#!training/Adam/gru/gru_cell/bias/m:MI
G
_user_specified_name/-training/Adam/gru/gru_cell/recurrent_kernel/m:C?
=
_user_specified_name%#training/Adam/gru/gru_cell/kernel/m::6
4
_user_specified_nametraining/Adam/dense/bias/m:<8
6
_user_specified_nametraining/Adam/dense/kernel/m:/+
)
_user_specified_namefalse_negatives:/+
)
_user_specified_namefalse_positives:.*
(
_user_specified_nametrue_negatives:.*
(
_user_specified_nametrue_positives:%!

_user_specified_namecount:%!

_user_specified_nametotal:;
7
5
_user_specified_nametraining/Adam/learning_rate:3	/
-
_user_specified_nametraining/Adam/decay:40
.
_user_specified_nametraining/Adam/beta_2:40
.
_user_specified_nametraining/Adam/beta_1:2.
,
_user_specified_nametraining/Adam/iter:1-
+
_user_specified_namegru/gru_cell/bias:=9
7
_user_specified_namegru/gru_cell/recurrent_kernel:3/
-
_user_specified_namegru/gru_cell/kernel:*&
$
_user_specified_name
dense/bias:,(
&
_user_specified_namedense/kernel:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�
�
B__inference_gru_cell_layer_call_and_return_conditional_losses_3393

inputs
states_02
 readvariableop_gru_gru_cell_bias:`;
)matmul_readvariableop_gru_gru_cell_kernel:`G
5matmul_1_readvariableop_gru_gru_cell_recurrent_kernel: `
identity

identity_1��MatMul/ReadVariableOp�MatMul_1/ReadVariableOp�ReadVariableOpo
ReadVariableOpReadVariableOp readvariableop_gru_gru_cell_bias*
_output_shapes

:`*
dtype0_
unstackUnpackReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
num
MatMul/ReadVariableOpReadVariableOp)matmul_readvariableop_gru_gru_cell_kernel*
_output_shapes

:`*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������`h
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*'
_output_shapes
:���������`Z
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*M
_output_shapes;
9:��������� :��������� :��������� *
	num_split�
MatMul_1/ReadVariableOpReadVariableOp5matmul_1_readvariableop_gru_gru_cell_recurrent_kernel*
_output_shapes

: `*
dtype0o
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������`l
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*'
_output_shapes
:���������`Z
ConstConst*
_output_shapes
:*
dtype0*!
valueB"        ����\
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
split_1SplitVBiasAdd_1:output:0Const:output:0split_1/split_dim:output:0*

Tlen0*
T0*M
_output_shapes;
9:��������� :��������� :��������� *
	num_split`
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:��������� M
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:��������� b
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:��������� Q
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:��������� ]
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:��������� Y
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:��������� I
TanhTanh	add_2:z:0*
T0*'
_output_shapes
:��������� U
mul_1MulSigmoid:y:0states_0*
T0*'
_output_shapes
:��������� J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:��������� Q
mul_2Mulsub:z:0Tanh:y:0*
T0*'
_output_shapes
:��������� V
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*'
_output_shapes
:��������� X
IdentityIdentity	add_3:z:0^NoOp*
T0*'
_output_shapes
:��������� Z

Identity_1Identity	add_3:z:0^NoOp*
T0*'
_output_shapes
:��������� e
NoOpNoOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
_output_shapes
 "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:���������:��������� : : : 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp:=9
7
_user_specified_namegru/gru_cell/recurrent_kernel:3/
-
_user_specified_namegru/gru_cell/kernel:1-
+
_user_specified_namegru/gru_cell/bias:QM
'
_output_shapes
:��������� 
"
_user_specified_name
states_0:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
g
=__inference_dot_layer_call_and_return_conditional_losses_2079

inputs
inputs_1
identityc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          p
	transpose	Transposeinputs_1transpose/perm:output:0*
T0*,
_output_shapes
:���������
�f
MatMulBatchMatMulV2inputstranspose:y:0*
T0*-
_output_shapes
:�����������R
ShapeShapeMatMul:output:0*
T0*
_output_shapes
::��]
IdentityIdentityMatMul:output:0*
T0*-
_output_shapes
:�����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:����������
:����������
:TP
,
_output_shapes
:����������

 
_user_specified_nameinputs:T P
,
_output_shapes
:����������

 
_user_specified_nameinputs
�
�
while_cond_1955
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
less_strided_slice_1,
(while_cond_1955___redundant_placeholder0,
(while_cond_1955___redundant_placeholder1,
(while_cond_1955___redundant_placeholder2,
(while_cond_1955___redundant_placeholder3
identity
P
LessLessplaceholderless_strided_slice_1*
T0*
_output_shapes
: ?
IdentityIdentityLess:z:0*
T0
*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :��������� : :::::

_output_shapes
::GC

_output_shapes
: 
)
_user_specified_namestrided_slice_1:-)
'
_output_shapes
:��������� :

_output_shapes
: :

_output_shapes
: :PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter
�
V
*__inference_concatenate_layer_call_fn_2616
inputs_0
inputs_1
identity�
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_concatenate_layer_call_and_return_conditional_losses_2108e
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:����������
:����������
:VR
,
_output_shapes
:����������

"
_user_specified_name
inputs_1:V R
,
_output_shapes
:����������

"
_user_specified_name
inputs_0
�
�
while_cond_3184
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
less_strided_slice_1,
(while_cond_3184___redundant_placeholder0,
(while_cond_3184___redundant_placeholder1,
(while_cond_3184___redundant_placeholder2,
(while_cond_3184___redundant_placeholder3
identity
P
LessLessplaceholderless_strided_slice_1*
T0*
_output_shapes
: ?
IdentityIdentityLess:z:0*
T0
*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :��������� : :::::

_output_shapes
::GC

_output_shapes
: 
)
_user_specified_namestrided_slice_1:-)
'
_output_shapes
:��������� :

_output_shapes
: :

_output_shapes
: :PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter
�
�
while_cond_2719
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
less_strided_slice_1,
(while_cond_2719___redundant_placeholder0,
(while_cond_2719___redundant_placeholder1,
(while_cond_2719___redundant_placeholder2,
(while_cond_2719___redundant_placeholder3
identity
P
LessLessplaceholderless_strided_slice_1*
T0*
_output_shapes
: ?
IdentityIdentityLess:z:0*
T0
*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :��������� : :::::

_output_shapes
::GC

_output_shapes
: 
)
_user_specified_namestrided_slice_1:-)
'
_output_shapes
:��������� :

_output_shapes
: :

_output_shapes
: :PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter
�
c
G__inference_masking_prior_layer_call_and_return_conditional_losses_2560

inputs
identityO

NotEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ��h
NotEqualNotEqualinputsNotEqual/y:output:0*
T0*,
_output_shapes
:����������
`
Any/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
���������w
AnyAnyNotEqual:z:0Any/reduction_indices:output:0*,
_output_shapes
:����������*
	keep_dims(`
CastCastAny:output:0*

DstT0*

SrcT0
*,
_output_shapes
:����������S
mulMulinputsCast:y:0*
T0*,
_output_shapes
:����������
s
SqueezeSqueezeAny:output:0*
T0
*(
_output_shapes
:����������*
squeeze_dims

���������T
IdentityIdentitymul:z:0*
T0*,
_output_shapes
:����������
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������
:T P
,
_output_shapes
:����������

 
_user_specified_nameinputs
�
�
$__inference_model_layer_call_fn_2476
input_prior#
gru_gru_cell_bias:`%
gru_gru_cell_kernel:`/
gru_gru_cell_recurrent_kernel: `
dense_kernel: 

dense_bias:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_priorgru_gru_cell_biasgru_gru_cell_kernelgru_gru_cell_recurrent_kerneldense_kernel
dense_bias*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8� *H
fCRA
?__inference_model_layer_call_and_return_conditional_losses_2456o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":����������
: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:*&
$
_user_specified_name
dense/bias:,(
&
_user_specified_namedense/kernel:=9
7
_user_specified_namegru/gru_cell/recurrent_kernel:3/
-
_user_specified_namegru/gru_cell/kernel:1-
+
_user_specified_namegru/gru_cell/bias:Y U
,
_output_shapes
:����������

%
_user_specified_nameinput_prior
�I
�	
model_gru_while_body_1630 
model_gru_while_loop_counter&
"model_gru_while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
model_gru_strided_slice_1_0[
Wtensorarrayv2read_tensorlistgetitem_model_gru_tensorarrayunstack_tensorlistfromtensor_0_
[tensorarrayv2read_1_tensorlistgetitem_model_gru_tensorarrayunstack_1_tensorlistfromtensor_0=
+gru_cell_readvariableop_gru_gru_cell_bias_0:`F
4gru_cell_matmul_readvariableop_gru_gru_cell_kernel_0:`R
@gru_cell_matmul_1_readvariableop_gru_gru_cell_recurrent_kernel_0: `
identity

identity_1

identity_2

identity_3

identity_4

identity_5
model_gru_strided_slice_1Y
Utensorarrayv2read_tensorlistgetitem_model_gru_tensorarrayunstack_tensorlistfromtensor]
Ytensorarrayv2read_1_tensorlistgetitem_model_gru_tensorarrayunstack_1_tensorlistfromtensor;
)gru_cell_readvariableop_gru_gru_cell_bias:`D
2gru_cell_matmul_readvariableop_gru_gru_cell_kernel:`P
>gru_cell_matmul_1_readvariableop_gru_gru_cell_recurrent_kernel: `��gru_cell/MatMul/ReadVariableOp� gru_cell/MatMul_1/ReadVariableOp�gru_cell/ReadVariableOp�
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
#TensorArrayV2Read/TensorListGetItemTensorListGetItemWtensorarrayv2read_tensorlistgetitem_model_gru_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
3TensorArrayV2Read_1/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
%TensorArrayV2Read_1/TensorListGetItemTensorListGetItem[tensorarrayv2read_1_tensorlistgetitem_model_gru_tensorarrayunstack_1_tensorlistfromtensor_0placeholder<TensorArrayV2Read_1/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0
�
gru_cell/ReadVariableOpReadVariableOp+gru_cell_readvariableop_gru_gru_cell_bias_0*
_output_shapes

:`*
dtype0q
gru_cell/unstackUnpackgru_cell/ReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
num�
gru_cell/MatMul/ReadVariableOpReadVariableOp4gru_cell_matmul_readvariableop_gru_gru_cell_kernel_0*
_output_shapes

:`*
dtype0�
gru_cell/MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0&gru_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������`�
gru_cell/BiasAddBiasAddgru_cell/MatMul:product:0gru_cell/unstack:output:0*
T0*'
_output_shapes
:���������`c
gru_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
gru_cell/splitSplit!gru_cell/split/split_dim:output:0gru_cell/BiasAdd:output:0*
T0*M
_output_shapes;
9:��������� :��������� :��������� *
	num_split�
 gru_cell/MatMul_1/ReadVariableOpReadVariableOp@gru_cell_matmul_1_readvariableop_gru_gru_cell_recurrent_kernel_0*
_output_shapes

: `*
dtype0�
gru_cell/MatMul_1MatMulplaceholder_3(gru_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������`�
gru_cell/BiasAdd_1BiasAddgru_cell/MatMul_1:product:0gru_cell/unstack:output:1*
T0*'
_output_shapes
:���������`c
gru_cell/ConstConst*
_output_shapes
:*
dtype0*!
valueB"        ����e
gru_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
gru_cell/split_1SplitVgru_cell/BiasAdd_1:output:0gru_cell/Const:output:0#gru_cell/split_1/split_dim:output:0*

Tlen0*
T0*M
_output_shapes;
9:��������� :��������� :��������� *
	num_split{
gru_cell/addAddV2gru_cell/split:output:0gru_cell/split_1:output:0*
T0*'
_output_shapes
:��������� _
gru_cell/SigmoidSigmoidgru_cell/add:z:0*
T0*'
_output_shapes
:��������� }
gru_cell/add_1AddV2gru_cell/split:output:1gru_cell/split_1:output:1*
T0*'
_output_shapes
:��������� c
gru_cell/Sigmoid_1Sigmoidgru_cell/add_1:z:0*
T0*'
_output_shapes
:��������� x
gru_cell/mulMulgru_cell/Sigmoid_1:y:0gru_cell/split_1:output:2*
T0*'
_output_shapes
:��������� t
gru_cell/add_2AddV2gru_cell/split:output:2gru_cell/mul:z:0*
T0*'
_output_shapes
:��������� [
gru_cell/TanhTanhgru_cell/add_2:z:0*
T0*'
_output_shapes
:��������� l
gru_cell/mul_1Mulgru_cell/Sigmoid:y:0placeholder_3*
T0*'
_output_shapes
:��������� S
gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?t
gru_cell/subSubgru_cell/sub/x:output:0gru_cell/Sigmoid:y:0*
T0*'
_output_shapes
:��������� l
gru_cell/mul_2Mulgru_cell/sub:z:0gru_cell/Tanh:y:0*
T0*'
_output_shapes
:��������� q
gru_cell/add_3AddV2gru_cell/mul_1:z:0gru_cell/mul_2:z:0*
T0*'
_output_shapes
:��������� _
Tile/multiplesConst*
_output_shapes
:*
dtype0*
valueB"      �
TileTile,TensorArrayV2Read_1/TensorListGetItem:item:0Tile/multiples:output:0*
T0
*'
_output_shapes
:���������x
SelectV2SelectV2Tile:output:0gru_cell/add_3:z:0placeholder_2*
T0*'
_output_shapes
:��������� a
Tile_1/multiplesConst*
_output_shapes
:*
dtype0*
valueB"      �
Tile_1Tile,TensorArrayV2Read_1/TensorListGetItem:item:0Tile_1/multiples:output:0*
T0
*'
_output_shapes
:���������|

SelectV2_1SelectV2Tile_1:output:0gru_cell/add_3:z:0placeholder_3*
T0*'
_output_shapes
:��������� l
*TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : �
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_13TensorArrayV2Write/TensorListSetItem/index:output:0SelectV2:output:0*
_output_shapes
: *
element_dtype0:���G
add/yConst*
_output_shapes
: *
dtype0*
value	B :J
addAddV2placeholderadd/y:output:0*
T0*
_output_shapes
: I
add_1/yConst*
_output_shapes
: *
dtype0*
value	B :_
add_1AddV2model_gru_while_loop_counteradd_1/y:output:0*
T0*
_output_shapes
: G
IdentityIdentity	add_1:z:0^NoOp*
T0*
_output_shapes
: b

Identity_1Identity"model_gru_while_maximum_iterations^NoOp*
T0*
_output_shapes
: G

Identity_2Identityadd:z:0^NoOp*
T0*
_output_shapes
: t

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^NoOp*
T0*
_output_shapes
: b

Identity_4IdentitySelectV2:output:0^NoOp*
T0*'
_output_shapes
:��������� d

Identity_5IdentitySelectV2_1:output:0^NoOp*
T0*'
_output_shapes
:��������� �
NoOpNoOp^gru_cell/MatMul/ReadVariableOp!^gru_cell/MatMul_1/ReadVariableOp^gru_cell/ReadVariableOp*
_output_shapes
 "�
>gru_cell_matmul_1_readvariableop_gru_gru_cell_recurrent_kernel@gru_cell_matmul_1_readvariableop_gru_gru_cell_recurrent_kernel_0"j
2gru_cell_matmul_readvariableop_gru_gru_cell_kernel4gru_cell_matmul_readvariableop_gru_gru_cell_kernel_0"X
)gru_cell_readvariableop_gru_gru_cell_bias+gru_cell_readvariableop_gru_gru_cell_bias_0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"
identityIdentity:output:0"8
model_gru_strided_slice_1model_gru_strided_slice_1_0"�
Ytensorarrayv2read_1_tensorlistgetitem_model_gru_tensorarrayunstack_1_tensorlistfromtensor[tensorarrayv2read_1_tensorlistgetitem_model_gru_tensorarrayunstack_1_tensorlistfromtensor_0"�
Utensorarrayv2read_tensorlistgetitem_model_gru_tensorarrayunstack_tensorlistfromtensorWtensorarrayv2read_tensorlistgetitem_model_gru_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :��������� :��������� : : : : : : 2@
gru_cell/MatMul/ReadVariableOpgru_cell/MatMul/ReadVariableOp2D
 gru_cell/MatMul_1/ReadVariableOp gru_cell/MatMul_1/ReadVariableOp22
gru_cell/ReadVariableOpgru_cell/ReadVariableOp:=9
7
_user_specified_namegru/gru_cell/recurrent_kernel:3
/
-
_user_specified_namegru/gru_cell/kernel:1	-
+
_user_specified_namegru/gru_cell/bias:kg

_output_shapes
: 
M
_user_specified_name53model/gru/TensorArrayUnstack_1/TensorListFromTensor:ie

_output_shapes
: 
K
_user_specified_name31model/gru/TensorArrayUnstack/TensorListFromTensor:QM

_output_shapes
: 
3
_user_specified_namemodel/gru/strided_slice_1:-)
'
_output_shapes
:��������� :-)
'
_output_shapes
:��������� :

_output_shapes
: :

_output_shapes
: :ZV

_output_shapes
: 
<
_user_specified_name$"model/gru/while/maximum_iterations:T P

_output_shapes
: 
6
_user_specified_namemodel/gru/while/loop_counter
��
�
__inference__traced_save_3571
file_prefix5
#read_disablecopyonread_dense_kernel: 1
#read_1_disablecopyonread_dense_bias:>
,read_2_disablecopyonread_gru_gru_cell_kernel:`H
6read_3_disablecopyonread_gru_gru_cell_recurrent_kernel: `<
*read_4_disablecopyonread_gru_gru_cell_bias:`5
+read_5_disablecopyonread_training_adam_iter:	 7
-read_6_disablecopyonread_training_adam_beta_1: 7
-read_7_disablecopyonread_training_adam_beta_2: 6
,read_8_disablecopyonread_training_adam_decay: >
4read_9_disablecopyonread_training_adam_learning_rate: )
read_10_disablecopyonread_total: )
read_11_disablecopyonread_count: 7
(read_12_disablecopyonread_true_positives:	�7
(read_13_disablecopyonread_true_negatives:	�8
)read_14_disablecopyonread_false_positives:	�8
)read_15_disablecopyonread_false_negatives:	�H
6read_16_disablecopyonread_training_adam_dense_kernel_m: B
4read_17_disablecopyonread_training_adam_dense_bias_m:O
=read_18_disablecopyonread_training_adam_gru_gru_cell_kernel_m:`Y
Gread_19_disablecopyonread_training_adam_gru_gru_cell_recurrent_kernel_m: `M
;read_20_disablecopyonread_training_adam_gru_gru_cell_bias_m:`H
6read_21_disablecopyonread_training_adam_dense_kernel_v: B
4read_22_disablecopyonread_training_adam_dense_bias_v:O
=read_23_disablecopyonread_training_adam_gru_gru_cell_kernel_v:`Y
Gread_24_disablecopyonread_training_adam_gru_gru_cell_recurrent_kernel_v: `M
;read_25_disablecopyonread_training_adam_gru_gru_cell_bias_v:`
savev2_const
identity_53��MergeV2Checkpoints�Read/DisableCopyOnRead�Read/ReadVariableOp�Read_1/DisableCopyOnRead�Read_1/ReadVariableOp�Read_10/DisableCopyOnRead�Read_10/ReadVariableOp�Read_11/DisableCopyOnRead�Read_11/ReadVariableOp�Read_12/DisableCopyOnRead�Read_12/ReadVariableOp�Read_13/DisableCopyOnRead�Read_13/ReadVariableOp�Read_14/DisableCopyOnRead�Read_14/ReadVariableOp�Read_15/DisableCopyOnRead�Read_15/ReadVariableOp�Read_16/DisableCopyOnRead�Read_16/ReadVariableOp�Read_17/DisableCopyOnRead�Read_17/ReadVariableOp�Read_18/DisableCopyOnRead�Read_18/ReadVariableOp�Read_19/DisableCopyOnRead�Read_19/ReadVariableOp�Read_2/DisableCopyOnRead�Read_2/ReadVariableOp�Read_20/DisableCopyOnRead�Read_20/ReadVariableOp�Read_21/DisableCopyOnRead�Read_21/ReadVariableOp�Read_22/DisableCopyOnRead�Read_22/ReadVariableOp�Read_23/DisableCopyOnRead�Read_23/ReadVariableOp�Read_24/DisableCopyOnRead�Read_24/ReadVariableOp�Read_25/DisableCopyOnRead�Read_25/ReadVariableOp�Read_3/DisableCopyOnRead�Read_3/ReadVariableOp�Read_4/DisableCopyOnRead�Read_4/ReadVariableOp�Read_5/DisableCopyOnRead�Read_5/ReadVariableOp�Read_6/DisableCopyOnRead�Read_6/ReadVariableOp�Read_7/DisableCopyOnRead�Read_7/ReadVariableOp�Read_8/DisableCopyOnRead�Read_8/ReadVariableOp�Read_9/DisableCopyOnRead�Read_9/ReadVariableOpw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: u
Read/DisableCopyOnReadDisableCopyOnRead#read_disablecopyonread_dense_kernel"/device:CPU:0*
_output_shapes
 �
Read/ReadVariableOpReadVariableOp#read_disablecopyonread_dense_kernel^Read/DisableCopyOnRead"/device:CPU:0*
_output_shapes

: *
dtype0i
IdentityIdentityRead/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

: a

Identity_1IdentityIdentity:output:0"/device:CPU:0*
T0*
_output_shapes

: w
Read_1/DisableCopyOnReadDisableCopyOnRead#read_1_disablecopyonread_dense_bias"/device:CPU:0*
_output_shapes
 �
Read_1/ReadVariableOpReadVariableOp#read_1_disablecopyonread_dense_bias^Read_1/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0i

Identity_2IdentityRead_1/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:_

Identity_3IdentityIdentity_2:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_2/DisableCopyOnReadDisableCopyOnRead,read_2_disablecopyonread_gru_gru_cell_kernel"/device:CPU:0*
_output_shapes
 �
Read_2/ReadVariableOpReadVariableOp,read_2_disablecopyonread_gru_gru_cell_kernel^Read_2/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:`*
dtype0m

Identity_4IdentityRead_2/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:`c

Identity_5IdentityIdentity_4:output:0"/device:CPU:0*
T0*
_output_shapes

:`�
Read_3/DisableCopyOnReadDisableCopyOnRead6read_3_disablecopyonread_gru_gru_cell_recurrent_kernel"/device:CPU:0*
_output_shapes
 �
Read_3/ReadVariableOpReadVariableOp6read_3_disablecopyonread_gru_gru_cell_recurrent_kernel^Read_3/DisableCopyOnRead"/device:CPU:0*
_output_shapes

: `*
dtype0m

Identity_6IdentityRead_3/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

: `c

Identity_7IdentityIdentity_6:output:0"/device:CPU:0*
T0*
_output_shapes

: `~
Read_4/DisableCopyOnReadDisableCopyOnRead*read_4_disablecopyonread_gru_gru_cell_bias"/device:CPU:0*
_output_shapes
 �
Read_4/ReadVariableOpReadVariableOp*read_4_disablecopyonread_gru_gru_cell_bias^Read_4/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:`*
dtype0m

Identity_8IdentityRead_4/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:`c

Identity_9IdentityIdentity_8:output:0"/device:CPU:0*
T0*
_output_shapes

:`
Read_5/DisableCopyOnReadDisableCopyOnRead+read_5_disablecopyonread_training_adam_iter"/device:CPU:0*
_output_shapes
 �
Read_5/ReadVariableOpReadVariableOp+read_5_disablecopyonread_training_adam_iter^Read_5/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0	f
Identity_10IdentityRead_5/ReadVariableOp:value:0"/device:CPU:0*
T0	*
_output_shapes
: ]
Identity_11IdentityIdentity_10:output:0"/device:CPU:0*
T0	*
_output_shapes
: �
Read_6/DisableCopyOnReadDisableCopyOnRead-read_6_disablecopyonread_training_adam_beta_1"/device:CPU:0*
_output_shapes
 �
Read_6/ReadVariableOpReadVariableOp-read_6_disablecopyonread_training_adam_beta_1^Read_6/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0f
Identity_12IdentityRead_6/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_13IdentityIdentity_12:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_7/DisableCopyOnReadDisableCopyOnRead-read_7_disablecopyonread_training_adam_beta_2"/device:CPU:0*
_output_shapes
 �
Read_7/ReadVariableOpReadVariableOp-read_7_disablecopyonread_training_adam_beta_2^Read_7/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0f
Identity_14IdentityRead_7/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_15IdentityIdentity_14:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_8/DisableCopyOnReadDisableCopyOnRead,read_8_disablecopyonread_training_adam_decay"/device:CPU:0*
_output_shapes
 �
Read_8/ReadVariableOpReadVariableOp,read_8_disablecopyonread_training_adam_decay^Read_8/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0f
Identity_16IdentityRead_8/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_17IdentityIdentity_16:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_9/DisableCopyOnReadDisableCopyOnRead4read_9_disablecopyonread_training_adam_learning_rate"/device:CPU:0*
_output_shapes
 �
Read_9/ReadVariableOpReadVariableOp4read_9_disablecopyonread_training_adam_learning_rate^Read_9/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0f
Identity_18IdentityRead_9/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_19IdentityIdentity_18:output:0"/device:CPU:0*
T0*
_output_shapes
: t
Read_10/DisableCopyOnReadDisableCopyOnReadread_10_disablecopyonread_total"/device:CPU:0*
_output_shapes
 �
Read_10/ReadVariableOpReadVariableOpread_10_disablecopyonread_total^Read_10/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_20IdentityRead_10/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_21IdentityIdentity_20:output:0"/device:CPU:0*
T0*
_output_shapes
: t
Read_11/DisableCopyOnReadDisableCopyOnReadread_11_disablecopyonread_count"/device:CPU:0*
_output_shapes
 �
Read_11/ReadVariableOpReadVariableOpread_11_disablecopyonread_count^Read_11/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_22IdentityRead_11/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_23IdentityIdentity_22:output:0"/device:CPU:0*
T0*
_output_shapes
: }
Read_12/DisableCopyOnReadDisableCopyOnRead(read_12_disablecopyonread_true_positives"/device:CPU:0*
_output_shapes
 �
Read_12/ReadVariableOpReadVariableOp(read_12_disablecopyonread_true_positives^Read_12/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_24IdentityRead_12/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_25IdentityIdentity_24:output:0"/device:CPU:0*
T0*
_output_shapes	
:�}
Read_13/DisableCopyOnReadDisableCopyOnRead(read_13_disablecopyonread_true_negatives"/device:CPU:0*
_output_shapes
 �
Read_13/ReadVariableOpReadVariableOp(read_13_disablecopyonread_true_negatives^Read_13/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_26IdentityRead_13/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_27IdentityIdentity_26:output:0"/device:CPU:0*
T0*
_output_shapes	
:�~
Read_14/DisableCopyOnReadDisableCopyOnRead)read_14_disablecopyonread_false_positives"/device:CPU:0*
_output_shapes
 �
Read_14/ReadVariableOpReadVariableOp)read_14_disablecopyonread_false_positives^Read_14/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_28IdentityRead_14/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_29IdentityIdentity_28:output:0"/device:CPU:0*
T0*
_output_shapes	
:�~
Read_15/DisableCopyOnReadDisableCopyOnRead)read_15_disablecopyonread_false_negatives"/device:CPU:0*
_output_shapes
 �
Read_15/ReadVariableOpReadVariableOp)read_15_disablecopyonread_false_negatives^Read_15/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_30IdentityRead_15/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_31IdentityIdentity_30:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_16/DisableCopyOnReadDisableCopyOnRead6read_16_disablecopyonread_training_adam_dense_kernel_m"/device:CPU:0*
_output_shapes
 �
Read_16/ReadVariableOpReadVariableOp6read_16_disablecopyonread_training_adam_dense_kernel_m^Read_16/DisableCopyOnRead"/device:CPU:0*
_output_shapes

: *
dtype0o
Identity_32IdentityRead_16/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

: e
Identity_33IdentityIdentity_32:output:0"/device:CPU:0*
T0*
_output_shapes

: �
Read_17/DisableCopyOnReadDisableCopyOnRead4read_17_disablecopyonread_training_adam_dense_bias_m"/device:CPU:0*
_output_shapes
 �
Read_17/ReadVariableOpReadVariableOp4read_17_disablecopyonread_training_adam_dense_bias_m^Read_17/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_34IdentityRead_17/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_35IdentityIdentity_34:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_18/DisableCopyOnReadDisableCopyOnRead=read_18_disablecopyonread_training_adam_gru_gru_cell_kernel_m"/device:CPU:0*
_output_shapes
 �
Read_18/ReadVariableOpReadVariableOp=read_18_disablecopyonread_training_adam_gru_gru_cell_kernel_m^Read_18/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:`*
dtype0o
Identity_36IdentityRead_18/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:`e
Identity_37IdentityIdentity_36:output:0"/device:CPU:0*
T0*
_output_shapes

:`�
Read_19/DisableCopyOnReadDisableCopyOnReadGread_19_disablecopyonread_training_adam_gru_gru_cell_recurrent_kernel_m"/device:CPU:0*
_output_shapes
 �
Read_19/ReadVariableOpReadVariableOpGread_19_disablecopyonread_training_adam_gru_gru_cell_recurrent_kernel_m^Read_19/DisableCopyOnRead"/device:CPU:0*
_output_shapes

: `*
dtype0o
Identity_38IdentityRead_19/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

: `e
Identity_39IdentityIdentity_38:output:0"/device:CPU:0*
T0*
_output_shapes

: `�
Read_20/DisableCopyOnReadDisableCopyOnRead;read_20_disablecopyonread_training_adam_gru_gru_cell_bias_m"/device:CPU:0*
_output_shapes
 �
Read_20/ReadVariableOpReadVariableOp;read_20_disablecopyonread_training_adam_gru_gru_cell_bias_m^Read_20/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:`*
dtype0o
Identity_40IdentityRead_20/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:`e
Identity_41IdentityIdentity_40:output:0"/device:CPU:0*
T0*
_output_shapes

:`�
Read_21/DisableCopyOnReadDisableCopyOnRead6read_21_disablecopyonread_training_adam_dense_kernel_v"/device:CPU:0*
_output_shapes
 �
Read_21/ReadVariableOpReadVariableOp6read_21_disablecopyonread_training_adam_dense_kernel_v^Read_21/DisableCopyOnRead"/device:CPU:0*
_output_shapes

: *
dtype0o
Identity_42IdentityRead_21/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

: e
Identity_43IdentityIdentity_42:output:0"/device:CPU:0*
T0*
_output_shapes

: �
Read_22/DisableCopyOnReadDisableCopyOnRead4read_22_disablecopyonread_training_adam_dense_bias_v"/device:CPU:0*
_output_shapes
 �
Read_22/ReadVariableOpReadVariableOp4read_22_disablecopyonread_training_adam_dense_bias_v^Read_22/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_44IdentityRead_22/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_45IdentityIdentity_44:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_23/DisableCopyOnReadDisableCopyOnRead=read_23_disablecopyonread_training_adam_gru_gru_cell_kernel_v"/device:CPU:0*
_output_shapes
 �
Read_23/ReadVariableOpReadVariableOp=read_23_disablecopyonread_training_adam_gru_gru_cell_kernel_v^Read_23/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:`*
dtype0o
Identity_46IdentityRead_23/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:`e
Identity_47IdentityIdentity_46:output:0"/device:CPU:0*
T0*
_output_shapes

:`�
Read_24/DisableCopyOnReadDisableCopyOnReadGread_24_disablecopyonread_training_adam_gru_gru_cell_recurrent_kernel_v"/device:CPU:0*
_output_shapes
 �
Read_24/ReadVariableOpReadVariableOpGread_24_disablecopyonread_training_adam_gru_gru_cell_recurrent_kernel_v^Read_24/DisableCopyOnRead"/device:CPU:0*
_output_shapes

: `*
dtype0o
Identity_48IdentityRead_24/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

: `e
Identity_49IdentityIdentity_48:output:0"/device:CPU:0*
T0*
_output_shapes

: `�
Read_25/DisableCopyOnReadDisableCopyOnRead;read_25_disablecopyonread_training_adam_gru_gru_cell_bias_v"/device:CPU:0*
_output_shapes
 �
Read_25/ReadVariableOpReadVariableOp;read_25_disablecopyonread_training_adam_gru_gru_cell_bias_v^Read_25/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:`*
dtype0o
Identity_50IdentityRead_25/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:`e
Identity_51IdentityIdentity_50:output:0"/device:CPU:0*
T0*
_output_shapes

:`�
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/1/true_positives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/1/true_negatives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/1/false_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/1/false_negatives/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*I
value@B>B B B B B B B B B B B B B B B B B B B B B B B B B B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Identity_1:output:0Identity_3:output:0Identity_5:output:0Identity_7:output:0Identity_9:output:0Identity_11:output:0Identity_13:output:0Identity_15:output:0Identity_17:output:0Identity_19:output:0Identity_21:output:0Identity_23:output:0Identity_25:output:0Identity_27:output:0Identity_29:output:0Identity_31:output:0Identity_33:output:0Identity_35:output:0Identity_37:output:0Identity_39:output:0Identity_41:output:0Identity_43:output:0Identity_45:output:0Identity_47:output:0Identity_49:output:0Identity_51:output:0savev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *)
dtypes
2	�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 i
Identity_52Identityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: U
Identity_53IdentityIdentity_52:output:0^NoOp*
T0*
_output_shapes
: �
NoOpNoOp^MergeV2Checkpoints^Read/DisableCopyOnRead^Read/ReadVariableOp^Read_1/DisableCopyOnRead^Read_1/ReadVariableOp^Read_10/DisableCopyOnRead^Read_10/ReadVariableOp^Read_11/DisableCopyOnRead^Read_11/ReadVariableOp^Read_12/DisableCopyOnRead^Read_12/ReadVariableOp^Read_13/DisableCopyOnRead^Read_13/ReadVariableOp^Read_14/DisableCopyOnRead^Read_14/ReadVariableOp^Read_15/DisableCopyOnRead^Read_15/ReadVariableOp^Read_16/DisableCopyOnRead^Read_16/ReadVariableOp^Read_17/DisableCopyOnRead^Read_17/ReadVariableOp^Read_18/DisableCopyOnRead^Read_18/ReadVariableOp^Read_19/DisableCopyOnRead^Read_19/ReadVariableOp^Read_2/DisableCopyOnRead^Read_2/ReadVariableOp^Read_20/DisableCopyOnRead^Read_20/ReadVariableOp^Read_21/DisableCopyOnRead^Read_21/ReadVariableOp^Read_22/DisableCopyOnRead^Read_22/ReadVariableOp^Read_23/DisableCopyOnRead^Read_23/ReadVariableOp^Read_24/DisableCopyOnRead^Read_24/ReadVariableOp^Read_25/DisableCopyOnRead^Read_25/ReadVariableOp^Read_3/DisableCopyOnRead^Read_3/ReadVariableOp^Read_4/DisableCopyOnRead^Read_4/ReadVariableOp^Read_5/DisableCopyOnRead^Read_5/ReadVariableOp^Read_6/DisableCopyOnRead^Read_6/ReadVariableOp^Read_7/DisableCopyOnRead^Read_7/ReadVariableOp^Read_8/DisableCopyOnRead^Read_8/ReadVariableOp^Read_9/DisableCopyOnRead^Read_9/ReadVariableOp*
_output_shapes
 "#
identity_53Identity_53:output:0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : : : : : : : : : : : : : : : : : : : : : : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints20
Read/DisableCopyOnReadRead/DisableCopyOnRead2*
Read/ReadVariableOpRead/ReadVariableOp24
Read_1/DisableCopyOnReadRead_1/DisableCopyOnRead2.
Read_1/ReadVariableOpRead_1/ReadVariableOp26
Read_10/DisableCopyOnReadRead_10/DisableCopyOnRead20
Read_10/ReadVariableOpRead_10/ReadVariableOp26
Read_11/DisableCopyOnReadRead_11/DisableCopyOnRead20
Read_11/ReadVariableOpRead_11/ReadVariableOp26
Read_12/DisableCopyOnReadRead_12/DisableCopyOnRead20
Read_12/ReadVariableOpRead_12/ReadVariableOp26
Read_13/DisableCopyOnReadRead_13/DisableCopyOnRead20
Read_13/ReadVariableOpRead_13/ReadVariableOp26
Read_14/DisableCopyOnReadRead_14/DisableCopyOnRead20
Read_14/ReadVariableOpRead_14/ReadVariableOp26
Read_15/DisableCopyOnReadRead_15/DisableCopyOnRead20
Read_15/ReadVariableOpRead_15/ReadVariableOp26
Read_16/DisableCopyOnReadRead_16/DisableCopyOnRead20
Read_16/ReadVariableOpRead_16/ReadVariableOp26
Read_17/DisableCopyOnReadRead_17/DisableCopyOnRead20
Read_17/ReadVariableOpRead_17/ReadVariableOp26
Read_18/DisableCopyOnReadRead_18/DisableCopyOnRead20
Read_18/ReadVariableOpRead_18/ReadVariableOp26
Read_19/DisableCopyOnReadRead_19/DisableCopyOnRead20
Read_19/ReadVariableOpRead_19/ReadVariableOp24
Read_2/DisableCopyOnReadRead_2/DisableCopyOnRead2.
Read_2/ReadVariableOpRead_2/ReadVariableOp26
Read_20/DisableCopyOnReadRead_20/DisableCopyOnRead20
Read_20/ReadVariableOpRead_20/ReadVariableOp26
Read_21/DisableCopyOnReadRead_21/DisableCopyOnRead20
Read_21/ReadVariableOpRead_21/ReadVariableOp26
Read_22/DisableCopyOnReadRead_22/DisableCopyOnRead20
Read_22/ReadVariableOpRead_22/ReadVariableOp26
Read_23/DisableCopyOnReadRead_23/DisableCopyOnRead20
Read_23/ReadVariableOpRead_23/ReadVariableOp26
Read_24/DisableCopyOnReadRead_24/DisableCopyOnRead20
Read_24/ReadVariableOpRead_24/ReadVariableOp26
Read_25/DisableCopyOnReadRead_25/DisableCopyOnRead20
Read_25/ReadVariableOpRead_25/ReadVariableOp24
Read_3/DisableCopyOnReadRead_3/DisableCopyOnRead2.
Read_3/ReadVariableOpRead_3/ReadVariableOp24
Read_4/DisableCopyOnReadRead_4/DisableCopyOnRead2.
Read_4/ReadVariableOpRead_4/ReadVariableOp24
Read_5/DisableCopyOnReadRead_5/DisableCopyOnRead2.
Read_5/ReadVariableOpRead_5/ReadVariableOp24
Read_6/DisableCopyOnReadRead_6/DisableCopyOnRead2.
Read_6/ReadVariableOpRead_6/ReadVariableOp24
Read_7/DisableCopyOnReadRead_7/DisableCopyOnRead2.
Read_7/ReadVariableOpRead_7/ReadVariableOp24
Read_8/DisableCopyOnReadRead_8/DisableCopyOnRead2.
Read_8/ReadVariableOpRead_8/ReadVariableOp24
Read_9/DisableCopyOnReadRead_9/DisableCopyOnRead2.
Read_9/ReadVariableOpRead_9/ReadVariableOp:=9

_output_shapes
: 

_user_specified_nameConst:A=
;
_user_specified_name#!training/Adam/gru/gru_cell/bias/v:MI
G
_user_specified_name/-training/Adam/gru/gru_cell/recurrent_kernel/v:C?
=
_user_specified_name%#training/Adam/gru/gru_cell/kernel/v::6
4
_user_specified_nametraining/Adam/dense/bias/v:<8
6
_user_specified_nametraining/Adam/dense/kernel/v:A=
;
_user_specified_name#!training/Adam/gru/gru_cell/bias/m:MI
G
_user_specified_name/-training/Adam/gru/gru_cell/recurrent_kernel/m:C?
=
_user_specified_name%#training/Adam/gru/gru_cell/kernel/m::6
4
_user_specified_nametraining/Adam/dense/bias/m:<8
6
_user_specified_nametraining/Adam/dense/kernel/m:/+
)
_user_specified_namefalse_negatives:/+
)
_user_specified_namefalse_positives:.*
(
_user_specified_nametrue_negatives:.*
(
_user_specified_nametrue_positives:%!

_user_specified_namecount:%!

_user_specified_nametotal:;
7
5
_user_specified_nametraining/Adam/learning_rate:3	/
-
_user_specified_nametraining/Adam/decay:40
.
_user_specified_nametraining/Adam/beta_2:40
.
_user_specified_nametraining/Adam/beta_1:2.
,
_user_specified_nametraining/Adam/iter:1-
+
_user_specified_namegru/gru_cell/bias:=9
7
_user_specified_namegru/gru_cell/recurrent_kernel:3/
-
_user_specified_namegru/gru_cell/kernel:*&
$
_user_specified_name
dense/bias:,(
&
_user_specified_namedense/kernel:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�N
�
=__inference_gru_layer_call_and_return_conditional_losses_2447

inputs;
)gru_cell_readvariableop_gru_gru_cell_bias:`D
2gru_cell_matmul_readvariableop_gru_gru_cell_kernel:`P
>gru_cell_matmul_1_readvariableop_gru_gru_cell_recurrent_kernel: `
identity��gru_cell/MatMul/ReadVariableOp� gru_cell/MatMul_1/ReadVariableOp�gru_cell/ReadVariableOp�whileI
ShapeShapeinputs*
T0*
_output_shapes
::��]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:��������� c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          n
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:����������R
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
::��_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask�
gru_cell/ReadVariableOpReadVariableOp)gru_cell_readvariableop_gru_gru_cell_bias*
_output_shapes

:`*
dtype0q
gru_cell/unstackUnpackgru_cell/ReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
num�
gru_cell/MatMul/ReadVariableOpReadVariableOp2gru_cell_matmul_readvariableop_gru_gru_cell_kernel*
_output_shapes

:`*
dtype0�
gru_cell/MatMulMatMulstrided_slice_2:output:0&gru_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������`�
gru_cell/BiasAddBiasAddgru_cell/MatMul:product:0gru_cell/unstack:output:0*
T0*'
_output_shapes
:���������`c
gru_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
gru_cell/splitSplit!gru_cell/split/split_dim:output:0gru_cell/BiasAdd:output:0*
T0*M
_output_shapes;
9:��������� :��������� :��������� *
	num_split�
 gru_cell/MatMul_1/ReadVariableOpReadVariableOp>gru_cell_matmul_1_readvariableop_gru_gru_cell_recurrent_kernel*
_output_shapes

: `*
dtype0�
gru_cell/MatMul_1MatMulzeros:output:0(gru_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������`�
gru_cell/BiasAdd_1BiasAddgru_cell/MatMul_1:product:0gru_cell/unstack:output:1*
T0*'
_output_shapes
:���������`c
gru_cell/ConstConst*
_output_shapes
:*
dtype0*!
valueB"        ����e
gru_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
gru_cell/split_1SplitVgru_cell/BiasAdd_1:output:0gru_cell/Const:output:0#gru_cell/split_1/split_dim:output:0*

Tlen0*
T0*M
_output_shapes;
9:��������� :��������� :��������� *
	num_split{
gru_cell/addAddV2gru_cell/split:output:0gru_cell/split_1:output:0*
T0*'
_output_shapes
:��������� _
gru_cell/SigmoidSigmoidgru_cell/add:z:0*
T0*'
_output_shapes
:��������� }
gru_cell/add_1AddV2gru_cell/split:output:1gru_cell/split_1:output:1*
T0*'
_output_shapes
:��������� c
gru_cell/Sigmoid_1Sigmoidgru_cell/add_1:z:0*
T0*'
_output_shapes
:��������� x
gru_cell/mulMulgru_cell/Sigmoid_1:y:0gru_cell/split_1:output:2*
T0*'
_output_shapes
:��������� t
gru_cell/add_2AddV2gru_cell/split:output:2gru_cell/mul:z:0*
T0*'
_output_shapes
:��������� [
gru_cell/TanhTanhgru_cell/add_2:z:0*
T0*'
_output_shapes
:��������� m
gru_cell/mul_1Mulgru_cell/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:��������� S
gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?t
gru_cell/subSubgru_cell/sub/x:output:0gru_cell/Sigmoid:y:0*
T0*'
_output_shapes
:��������� l
gru_cell/mul_2Mulgru_cell/sub:z:0gru_cell/Tanh:y:0*
T0*'
_output_shapes
:��������� q
gru_cell/add_3AddV2gru_cell/mul_1:z:0gru_cell/mul_2:z:0*
T0*'
_output_shapes
:��������� n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :�
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)gru_cell_readvariableop_gru_gru_cell_bias2gru_cell_matmul_readvariableop_gru_gru_cell_kernel>gru_cell_matmul_1_readvariableop_gru_gru_cell_recurrent_kernel*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :��������� : : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_2357*
condR
while_cond_2356*8
output_shapes'
%: : : : :��������� : : : : : *
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:��������� *
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:��������� *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:��������� [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:��������� �
NoOpNoOp^gru_cell/MatMul/ReadVariableOp!^gru_cell/MatMul_1/ReadVariableOp^gru_cell/ReadVariableOp^while*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:����������: : : 2@
gru_cell/MatMul/ReadVariableOpgru_cell/MatMul/ReadVariableOp2D
 gru_cell/MatMul_1/ReadVariableOp gru_cell/MatMul_1/ReadVariableOp22
gru_cell/ReadVariableOpgru_cell/ReadVariableOp2
whilewhile:=9
7
_user_specified_namegru/gru_cell/recurrent_kernel:3/
-
_user_specified_namegru/gru_cell/kernel:1-
+
_user_specified_namegru/gru_cell/bias:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�9
�
while_body_2174
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
strided_slice_1_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0=
+gru_cell_readvariableop_gru_gru_cell_bias_0:`F
4gru_cell_matmul_readvariableop_gru_gru_cell_kernel_0:`R
@gru_cell_matmul_1_readvariableop_gru_gru_cell_recurrent_kernel_0: `
identity

identity_1

identity_2

identity_3

identity_4
strided_slice_1O
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor;
)gru_cell_readvariableop_gru_gru_cell_bias:`D
2gru_cell_matmul_readvariableop_gru_gru_cell_kernel:`P
>gru_cell_matmul_1_readvariableop_gru_gru_cell_recurrent_kernel: `��gru_cell/MatMul/ReadVariableOp� gru_cell/MatMul_1/ReadVariableOp�gru_cell/ReadVariableOp�
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
gru_cell/ReadVariableOpReadVariableOp+gru_cell_readvariableop_gru_gru_cell_bias_0*
_output_shapes

:`*
dtype0q
gru_cell/unstackUnpackgru_cell/ReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
num�
gru_cell/MatMul/ReadVariableOpReadVariableOp4gru_cell_matmul_readvariableop_gru_gru_cell_kernel_0*
_output_shapes

:`*
dtype0�
gru_cell/MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0&gru_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������`�
gru_cell/BiasAddBiasAddgru_cell/MatMul:product:0gru_cell/unstack:output:0*
T0*'
_output_shapes
:���������`c
gru_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
gru_cell/splitSplit!gru_cell/split/split_dim:output:0gru_cell/BiasAdd:output:0*
T0*M
_output_shapes;
9:��������� :��������� :��������� *
	num_split�
 gru_cell/MatMul_1/ReadVariableOpReadVariableOp@gru_cell_matmul_1_readvariableop_gru_gru_cell_recurrent_kernel_0*
_output_shapes

: `*
dtype0�
gru_cell/MatMul_1MatMulplaceholder_2(gru_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������`�
gru_cell/BiasAdd_1BiasAddgru_cell/MatMul_1:product:0gru_cell/unstack:output:1*
T0*'
_output_shapes
:���������`c
gru_cell/ConstConst*
_output_shapes
:*
dtype0*!
valueB"        ����e
gru_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
gru_cell/split_1SplitVgru_cell/BiasAdd_1:output:0gru_cell/Const:output:0#gru_cell/split_1/split_dim:output:0*

Tlen0*
T0*M
_output_shapes;
9:��������� :��������� :��������� *
	num_split{
gru_cell/addAddV2gru_cell/split:output:0gru_cell/split_1:output:0*
T0*'
_output_shapes
:��������� _
gru_cell/SigmoidSigmoidgru_cell/add:z:0*
T0*'
_output_shapes
:��������� }
gru_cell/add_1AddV2gru_cell/split:output:1gru_cell/split_1:output:1*
T0*'
_output_shapes
:��������� c
gru_cell/Sigmoid_1Sigmoidgru_cell/add_1:z:0*
T0*'
_output_shapes
:��������� x
gru_cell/mulMulgru_cell/Sigmoid_1:y:0gru_cell/split_1:output:2*
T0*'
_output_shapes
:��������� t
gru_cell/add_2AddV2gru_cell/split:output:2gru_cell/mul:z:0*
T0*'
_output_shapes
:��������� [
gru_cell/TanhTanhgru_cell/add_2:z:0*
T0*'
_output_shapes
:��������� l
gru_cell/mul_1Mulgru_cell/Sigmoid:y:0placeholder_2*
T0*'
_output_shapes
:��������� S
gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?t
gru_cell/subSubgru_cell/sub/x:output:0gru_cell/Sigmoid:y:0*
T0*'
_output_shapes
:��������� l
gru_cell/mul_2Mulgru_cell/sub:z:0gru_cell/Tanh:y:0*
T0*'
_output_shapes
:��������� q
gru_cell/add_3AddV2gru_cell/mul_1:z:0gru_cell/mul_2:z:0*
T0*'
_output_shapes
:��������� l
*TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : �
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_13TensorArrayV2Write/TensorListSetItem/index:output:0gru_cell/add_3:z:0*
_output_shapes
: *
element_dtype0:���G
add/yConst*
_output_shapes
: *
dtype0*
value	B :J
addAddV2placeholderadd/y:output:0*
T0*
_output_shapes
: I
add_1/yConst*
_output_shapes
: *
dtype0*
value	B :U
add_1AddV2while_loop_counteradd_1/y:output:0*
T0*
_output_shapes
: G
IdentityIdentity	add_1:z:0^NoOp*
T0*
_output_shapes
: X

Identity_1Identitywhile_maximum_iterations^NoOp*
T0*
_output_shapes
: G

Identity_2Identityadd:z:0^NoOp*
T0*
_output_shapes
: t

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^NoOp*
T0*
_output_shapes
: c

Identity_4Identitygru_cell/add_3:z:0^NoOp*
T0*'
_output_shapes
:��������� �
NoOpNoOp^gru_cell/MatMul/ReadVariableOp!^gru_cell/MatMul_1/ReadVariableOp^gru_cell/ReadVariableOp*
_output_shapes
 "�
>gru_cell_matmul_1_readvariableop_gru_gru_cell_recurrent_kernel@gru_cell_matmul_1_readvariableop_gru_gru_cell_recurrent_kernel_0"j
2gru_cell_matmul_readvariableop_gru_gru_cell_kernel4gru_cell_matmul_readvariableop_gru_gru_cell_kernel_0"X
)gru_cell_readvariableop_gru_gru_cell_bias+gru_cell_readvariableop_gru_gru_cell_bias_0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
identityIdentity:output:0"$
strided_slice_1strided_slice_1_0"�
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :��������� : : : : : 2@
gru_cell/MatMul/ReadVariableOpgru_cell/MatMul/ReadVariableOp2D
 gru_cell/MatMul_1/ReadVariableOp gru_cell/MatMul_1/ReadVariableOp22
gru_cell/ReadVariableOpgru_cell/ReadVariableOp:=	9
7
_user_specified_namegru/gru_cell/recurrent_kernel:3/
-
_user_specified_namegru/gru_cell/kernel:1-
+
_user_specified_namegru/gru_cell/bias:_[

_output_shapes
: 
A
_user_specified_name)'TensorArrayUnstack/TensorListFromTensor:GC

_output_shapes
: 
)
_user_specified_namestrided_slice_1:-)
'
_output_shapes
:��������� :

_output_shapes
: :

_output_shapes
: :PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter
�
�
while_cond_2356
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
less_strided_slice_1,
(while_cond_2356___redundant_placeholder0,
(while_cond_2356___redundant_placeholder1,
(while_cond_2356___redundant_placeholder2,
(while_cond_2356___redundant_placeholder3
identity
P
LessLessplaceholderless_strided_slice_1*
T0*
_output_shapes
: ?
IdentityIdentityLess:z:0*
T0
*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :��������� : :::::

_output_shapes
::GC

_output_shapes
: 
)
_user_specified_namestrided_slice_1:-)
'
_output_shapes
:��������� :

_output_shapes
: :

_output_shapes
: :PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter
�
�
while_cond_2173
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
less_strided_slice_1,
(while_cond_2173___redundant_placeholder0,
(while_cond_2173___redundant_placeholder1,
(while_cond_2173___redundant_placeholder2,
(while_cond_2173___redundant_placeholder3
identity
P
LessLessplaceholderless_strided_slice_1*
T0*
_output_shapes
: ?
IdentityIdentityLess:z:0*
T0
*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :��������� : :::::

_output_shapes
::GC

_output_shapes
: 
)
_user_specified_namestrided_slice_1:-)
'
_output_shapes
:��������� :

_output_shapes
: :

_output_shapes
: :PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter
�N
�
=__inference_gru_layer_call_and_return_conditional_losses_2810
inputs_0;
)gru_cell_readvariableop_gru_gru_cell_bias:`D
2gru_cell_matmul_readvariableop_gru_gru_cell_kernel:`P
>gru_cell_matmul_1_readvariableop_gru_gru_cell_recurrent_kernel: `
identity��gru_cell/MatMul/ReadVariableOp� gru_cell/MatMul_1/ReadVariableOp�gru_cell/ReadVariableOp�whileK
ShapeShapeinputs_0*
T0*
_output_shapes
::��]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:��������� c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :������������������R
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
::��_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask�
gru_cell/ReadVariableOpReadVariableOp)gru_cell_readvariableop_gru_gru_cell_bias*
_output_shapes

:`*
dtype0q
gru_cell/unstackUnpackgru_cell/ReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
num�
gru_cell/MatMul/ReadVariableOpReadVariableOp2gru_cell_matmul_readvariableop_gru_gru_cell_kernel*
_output_shapes

:`*
dtype0�
gru_cell/MatMulMatMulstrided_slice_2:output:0&gru_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������`�
gru_cell/BiasAddBiasAddgru_cell/MatMul:product:0gru_cell/unstack:output:0*
T0*'
_output_shapes
:���������`c
gru_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
gru_cell/splitSplit!gru_cell/split/split_dim:output:0gru_cell/BiasAdd:output:0*
T0*M
_output_shapes;
9:��������� :��������� :��������� *
	num_split�
 gru_cell/MatMul_1/ReadVariableOpReadVariableOp>gru_cell_matmul_1_readvariableop_gru_gru_cell_recurrent_kernel*
_output_shapes

: `*
dtype0�
gru_cell/MatMul_1MatMulzeros:output:0(gru_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������`�
gru_cell/BiasAdd_1BiasAddgru_cell/MatMul_1:product:0gru_cell/unstack:output:1*
T0*'
_output_shapes
:���������`c
gru_cell/ConstConst*
_output_shapes
:*
dtype0*!
valueB"        ����e
gru_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
gru_cell/split_1SplitVgru_cell/BiasAdd_1:output:0gru_cell/Const:output:0#gru_cell/split_1/split_dim:output:0*

Tlen0*
T0*M
_output_shapes;
9:��������� :��������� :��������� *
	num_split{
gru_cell/addAddV2gru_cell/split:output:0gru_cell/split_1:output:0*
T0*'
_output_shapes
:��������� _
gru_cell/SigmoidSigmoidgru_cell/add:z:0*
T0*'
_output_shapes
:��������� }
gru_cell/add_1AddV2gru_cell/split:output:1gru_cell/split_1:output:1*
T0*'
_output_shapes
:��������� c
gru_cell/Sigmoid_1Sigmoidgru_cell/add_1:z:0*
T0*'
_output_shapes
:��������� x
gru_cell/mulMulgru_cell/Sigmoid_1:y:0gru_cell/split_1:output:2*
T0*'
_output_shapes
:��������� t
gru_cell/add_2AddV2gru_cell/split:output:2gru_cell/mul:z:0*
T0*'
_output_shapes
:��������� [
gru_cell/TanhTanhgru_cell/add_2:z:0*
T0*'
_output_shapes
:��������� m
gru_cell/mul_1Mulgru_cell/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:��������� S
gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?t
gru_cell/subSubgru_cell/sub/x:output:0gru_cell/Sigmoid:y:0*
T0*'
_output_shapes
:��������� l
gru_cell/mul_2Mulgru_cell/sub:z:0gru_cell/Tanh:y:0*
T0*'
_output_shapes
:��������� q
gru_cell/add_3AddV2gru_cell/mul_1:z:0gru_cell/mul_2:z:0*
T0*'
_output_shapes
:��������� n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :�
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)gru_cell_readvariableop_gru_gru_cell_bias2gru_cell_matmul_readvariableop_gru_gru_cell_kernel>gru_cell_matmul_1_readvariableop_gru_gru_cell_recurrent_kernel*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :��������� : : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_2720*
condR
while_cond_2719*8
output_shapes'
%: : : : :��������� : : : : : *
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:��������� *
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:��������� *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:��������� [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:��������� �
NoOpNoOp^gru_cell/MatMul/ReadVariableOp!^gru_cell/MatMul_1/ReadVariableOp^gru_cell/ReadVariableOp^while*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������: : : 2@
gru_cell/MatMul/ReadVariableOpgru_cell/MatMul/ReadVariableOp2D
 gru_cell/MatMul_1/ReadVariableOp gru_cell/MatMul_1/ReadVariableOp22
gru_cell/ReadVariableOpgru_cell/ReadVariableOp2
whilewhile:=9
7
_user_specified_namegru/gru_cell/recurrent_kernel:3/
-
_user_specified_namegru/gru_cell/kernel:1-
+
_user_specified_namegru/gru_cell/bias:^ Z
4
_output_shapes"
 :������������������
"
_user_specified_name
inputs_0
�	
�
"__inference_gru_layer_call_fn_2639
inputs_0#
gru_gru_cell_bias:`%
gru_gru_cell_kernel:`/
gru_gru_cell_recurrent_kernel: `
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0gru_gru_cell_biasgru_gru_cell_kernelgru_gru_cell_recurrent_kernel*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *F
fAR?
=__inference_gru_layer_call_and_return_conditional_losses_2018o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:��������� <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������: : : 22
StatefulPartitionedCallStatefulPartitionedCall:=9
7
_user_specified_namegru/gru_cell/recurrent_kernel:3/
-
_user_specified_namegru/gru_cell/kernel:1-
+
_user_specified_namegru/gru_cell/bias:^ Z
4
_output_shapes"
 :������������������
"
_user_specified_name
inputs_0"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
H
input_prior9
serving_default_input_prior:0����������
9
dense0
StatefulPartitionedCall:0���������tensorflow/serving/predict:��
�
layer-0
layer-1
layer-2
layer-3
layer-4
layer-5
layer-6
layer_with_weights-0
layer-7
	layer_with_weights-1
	layer-8

	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures"
_tf_keras_network
"
_tf_keras_input_layer
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
axes"
_tf_keras_layer
�
 	variables
!trainable_variables
"regularization_losses
#	keras_api
$__call__
*%&call_and_return_all_conditional_losses"
_tf_keras_layer
�
&	variables
'trainable_variables
(regularization_losses
)	keras_api
*__call__
*+&call_and_return_all_conditional_losses
,axes"
_tf_keras_layer
�
-	variables
.trainable_variables
/regularization_losses
0	keras_api
1__call__
*2&call_and_return_all_conditional_losses"
_tf_keras_layer
�
3	variables
4trainable_variables
5regularization_losses
6	keras_api
7__call__
*8&call_and_return_all_conditional_losses"
_tf_keras_layer
�
9	variables
:trainable_variables
;regularization_losses
<	keras_api
=__call__
*>&call_and_return_all_conditional_losses
?_random_generator
@cell
A
state_spec"
_tf_keras_rnn_layer
�
B	variables
Ctrainable_variables
Dregularization_losses
E	keras_api
F__call__
*G&call_and_return_all_conditional_losses

Hkernel
Ibias"
_tf_keras_layer
C
J0
K1
L2
H3
I4"
trackable_list_wrapper
C
J0
K1
L2
H3
I4"
trackable_list_wrapper
 "
trackable_list_wrapper
�
Mnon_trainable_variables

Nlayers
Ometrics
Player_regularization_losses
Qlayer_metrics

	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
Rtrace_0
Strace_12�
$__inference_model_layer_call_fn_2466
$__inference_model_layer_call_fn_2476�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zRtrace_0zStrace_1
�
Ttrace_0
Utrace_12�
?__inference_model_layer_call_and_return_conditional_losses_2284
?__inference_model_layer_call_and_return_conditional_losses_2456�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zTtrace_0zUtrace_1
�B�
__inference__wrapped_model_1742input_prior"�
���
FullArgSpec
args�

jargs_0
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�
Viter

Wbeta_1

Xbeta_2
	Ydecay
Zlearning_rateHm�Im�Jm�Km�Lm�Hv�Iv�Jv�Kv�Lv�"
	optimizer
,
[serving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
\non_trainable_variables

]layers
^metrics
_layer_regularization_losses
`layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
atrace_02�
,__inference_masking_prior_layer_call_fn_2549�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zatrace_0
�
btrace_02�
G__inference_masking_prior_layer_call_and_return_conditional_losses_2560�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zbtrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
cnon_trainable_variables

dlayers
emetrics
flayer_regularization_losses
glayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
htrace_02�
"__inference_dot_layer_call_fn_2566�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zhtrace_0
�
itrace_02�
=__inference_dot_layer_call_and_return_conditional_losses_2575�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zitrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
jnon_trainable_variables

klayers
lmetrics
mlayer_regularization_losses
nlayer_metrics
 	variables
!trainable_variables
"regularization_losses
$__call__
*%&call_and_return_all_conditional_losses
&%"call_and_return_conditional_losses"
_generic_user_object
�
otrace_02�
)__inference_activation_layer_call_fn_2580�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zotrace_0
�
ptrace_02�
D__inference_activation_layer_call_and_return_conditional_losses_2585�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zptrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
qnon_trainable_variables

rlayers
smetrics
tlayer_regularization_losses
ulayer_metrics
&	variables
'trainable_variables
(regularization_losses
*__call__
*+&call_and_return_all_conditional_losses
&+"call_and_return_conditional_losses"
_generic_user_object
�
vtrace_02�
$__inference_dot_1_layer_call_fn_2591�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zvtrace_0
�
wtrace_02�
?__inference_dot_1_layer_call_and_return_conditional_losses_2598�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zwtrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
xnon_trainable_variables

ylayers
zmetrics
{layer_regularization_losses
|layer_metrics
-	variables
.trainable_variables
/regularization_losses
1__call__
*2&call_and_return_all_conditional_losses
&2"call_and_return_conditional_losses"
_generic_user_object
�
}trace_02�
'__inference_multiply_layer_call_fn_2604�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z}trace_0
�
~trace_02�
B__inference_multiply_layer_call_and_return_conditional_losses_2610�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z~trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
3	variables
4trainable_variables
5regularization_losses
7__call__
*8&call_and_return_all_conditional_losses
&8"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
*__inference_concatenate_layer_call_fn_2616�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
E__inference_concatenate_layer_call_and_return_conditional_losses_2623�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
5
J0
K1
L2"
trackable_list_wrapper
5
J0
K1
L2"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�states
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
9	variables
:trainable_variables
;regularization_losses
=__call__
*>&call_and_return_all_conditional_losses
&>"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_1
�trace_2
�trace_32�
"__inference_gru_layer_call_fn_2631
"__inference_gru_layer_call_fn_2639
"__inference_gru_layer_call_fn_2647
"__inference_gru_layer_call_fn_2655�
���
FullArgSpec:
args2�/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults�

 
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1z�trace_2z�trace_3
�
�trace_0
�trace_1
�trace_2
�trace_32�
=__inference_gru_layer_call_and_return_conditional_losses_2810
=__inference_gru_layer_call_and_return_conditional_losses_2965
=__inference_gru_layer_call_and_return_conditional_losses_3120
=__inference_gru_layer_call_and_return_conditional_losses_3275�
���
FullArgSpec:
args2�/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults�

 
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1z�trace_2z�trace_3
"
_generic_user_object
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�_random_generator

Jkernel
Krecurrent_kernel
Lbias"
_tf_keras_layer
 "
trackable_list_wrapper
.
H0
I1"
trackable_list_wrapper
.
H0
I1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
B	variables
Ctrainable_variables
Dregularization_losses
F__call__
*G&call_and_return_all_conditional_losses
&G"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
$__inference_dense_layer_call_fn_3282�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
?__inference_dense_layer_call_and_return_conditional_losses_3293�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
: 2dense/kernel
:2
dense/bias
%:#`2gru/gru_cell/kernel
/:- `2gru/gru_cell/recurrent_kernel
#:!`2gru/gru_cell/bias
 "
trackable_list_wrapper
_
0
1
2
3
4
5
6
7
	8"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
$__inference_model_layer_call_fn_2466input_prior"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
$__inference_model_layer_call_fn_2476input_prior"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
?__inference_model_layer_call_and_return_conditional_losses_2284input_prior"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
?__inference_model_layer_call_and_return_conditional_losses_2456input_prior"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
:	 (2training/Adam/iter
: (2training/Adam/beta_1
: (2training/Adam/beta_2
: (2training/Adam/decay
%:# (2training/Adam/learning_rate
�B�
"__inference_signature_wrapper_2544input_prior"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
  

kwonlyargs�
jinput_prior
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
,__inference_masking_prior_layer_call_fn_2549inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
G__inference_masking_prior_layer_call_and_return_conditional_losses_2560inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
"__inference_dot_layer_call_fn_2566inputs_0inputs_1"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
=__inference_dot_layer_call_and_return_conditional_losses_2575inputs_0inputs_1"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
)__inference_activation_layer_call_fn_2580inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
D__inference_activation_layer_call_and_return_conditional_losses_2585inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
$__inference_dot_1_layer_call_fn_2591inputs_0inputs_1"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
?__inference_dot_1_layer_call_and_return_conditional_losses_2598inputs_0inputs_1"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
'__inference_multiply_layer_call_fn_2604inputs_0inputs_1"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
B__inference_multiply_layer_call_and_return_conditional_losses_2610inputs_0inputs_1"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
*__inference_concatenate_layer_call_fn_2616inputs_0inputs_1"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
E__inference_concatenate_layer_call_and_return_conditional_losses_2623inputs_0inputs_1"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
@0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
"__inference_gru_layer_call_fn_2631inputs_0"�
���
FullArgSpec:
args2�/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
"__inference_gru_layer_call_fn_2639inputs_0"�
���
FullArgSpec:
args2�/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
"__inference_gru_layer_call_fn_2647inputs"�
���
FullArgSpec:
args2�/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
"__inference_gru_layer_call_fn_2655inputs"�
���
FullArgSpec:
args2�/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
=__inference_gru_layer_call_and_return_conditional_losses_2810inputs_0"�
���
FullArgSpec:
args2�/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
=__inference_gru_layer_call_and_return_conditional_losses_2965inputs_0"�
���
FullArgSpec:
args2�/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
=__inference_gru_layer_call_and_return_conditional_losses_3120inputs"�
���
FullArgSpec:
args2�/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
=__inference_gru_layer_call_and_return_conditional_losses_3275inputs"�
���
FullArgSpec:
args2�/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
5
J0
K1
L2"
trackable_list_wrapper
5
J0
K1
L2"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
'__inference_gru_cell_layer_call_fn_3304
'__inference_gru_cell_layer_call_fn_3315�
���
FullArgSpec+
args#� 
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
B__inference_gru_cell_layer_call_and_return_conditional_losses_3354
B__inference_gru_cell_layer_call_and_return_conditional_losses_3393�
���
FullArgSpec+
args#� 
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
$__inference_dense_layer_call_fn_3282inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
?__inference_dense_layer_call_and_return_conditional_losses_3293inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
c
�	variables
�	keras_api

�total

�count
�
_fn_kwargs"
_tf_keras_metric
�
�	variables
�	keras_api
�true_positives
�true_negatives
�false_positives
�false_negatives"
_tf_keras_metric
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
'__inference_gru_cell_layer_call_fn_3304inputsstates_0"�
���
FullArgSpec+
args#� 
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
'__inference_gru_cell_layer_call_fn_3315inputsstates_0"�
���
FullArgSpec+
args#� 
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
B__inference_gru_cell_layer_call_and_return_conditional_losses_3354inputsstates_0"�
���
FullArgSpec+
args#� 
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
B__inference_gru_cell_layer_call_and_return_conditional_losses_3393inputsstates_0"�
���
FullArgSpec+
args#� 
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
@
�0
�1
�2
�3"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:� (2true_positives
:� (2true_negatives
 :� (2false_positives
 :� (2false_negatives
,:* 2training/Adam/dense/kernel/m
&:$2training/Adam/dense/bias/m
3:1`2#training/Adam/gru/gru_cell/kernel/m
=:; `2-training/Adam/gru/gru_cell/recurrent_kernel/m
1:/`2!training/Adam/gru/gru_cell/bias/m
,:* 2training/Adam/dense/kernel/v
&:$2training/Adam/dense/bias/v
3:1`2#training/Adam/gru/gru_cell/kernel/v
=:; `2-training/Adam/gru/gru_cell/recurrent_kernel/v
1:/`2!training/Adam/gru/gru_cell/bias/v�
__inference__wrapped_model_1742qLJKHI9�6
/�,
*�'
input_prior����������

� "-�*
(
dense�
dense����������
D__inference_activation_layer_call_and_return_conditional_losses_2585k5�2
+�(
&�#
inputs�����������
� "2�/
(�%
tensor_0�����������
� �
)__inference_activation_layer_call_fn_2580`5�2
+�(
&�#
inputs�����������
� "'�$
unknown������������
E__inference_concatenate_layer_call_and_return_conditional_losses_2623�d�a
Z�W
U�R
'�$
inputs_0����������

'�$
inputs_1����������

� "1�.
'�$
tensor_0����������
� �
*__inference_concatenate_layer_call_fn_2616�d�a
Z�W
U�R
'�$
inputs_0����������

'�$
inputs_1����������

� "&�#
unknown�����������
?__inference_dense_layer_call_and_return_conditional_losses_3293cHI/�,
%�"
 �
inputs��������� 
� ",�)
"�
tensor_0���������
� �
$__inference_dense_layer_call_fn_3282XHI/�,
%�"
 �
inputs��������� 
� "!�
unknown����������
?__inference_dot_1_layer_call_and_return_conditional_losses_2598�e�b
[�X
V�S
(�%
inputs_0�����������
'�$
inputs_1����������

� "1�.
'�$
tensor_0����������

� �
$__inference_dot_1_layer_call_fn_2591�e�b
[�X
V�S
(�%
inputs_0�����������
'�$
inputs_1����������

� "&�#
unknown����������
�
=__inference_dot_layer_call_and_return_conditional_losses_2575�d�a
Z�W
U�R
'�$
inputs_0����������

'�$
inputs_1����������

� "2�/
(�%
tensor_0�����������
� �
"__inference_dot_layer_call_fn_2566�d�a
Z�W
U�R
'�$
inputs_0����������

'�$
inputs_1����������

� "'�$
unknown������������
B__inference_gru_cell_layer_call_and_return_conditional_losses_3354�LJK\�Y
R�O
 �
inputs���������
'�$
"�
states_0��������� 
p
� "`�]
V�S
$�!

tensor_0_0��������� 
+�(
&�#
tensor_0_1_0��������� 
� �
B__inference_gru_cell_layer_call_and_return_conditional_losses_3393�LJK\�Y
R�O
 �
inputs���������
'�$
"�
states_0��������� 
p 
� "`�]
V�S
$�!

tensor_0_0��������� 
+�(
&�#
tensor_0_1_0��������� 
� �
'__inference_gru_cell_layer_call_fn_3304�LJK\�Y
R�O
 �
inputs���������
'�$
"�
states_0��������� 
p
� "R�O
"�
tensor_0��������� 
)�&
$�!

tensor_1_0��������� �
'__inference_gru_cell_layer_call_fn_3315�LJK\�Y
R�O
 �
inputs���������
'�$
"�
states_0��������� 
p 
� "R�O
"�
tensor_0��������� 
)�&
$�!

tensor_1_0��������� �
=__inference_gru_layer_call_and_return_conditional_losses_2810�LJKO�L
E�B
4�1
/�,
inputs_0������������������

 
p

 
� ",�)
"�
tensor_0��������� 
� �
=__inference_gru_layer_call_and_return_conditional_losses_2965�LJKO�L
E�B
4�1
/�,
inputs_0������������������

 
p 

 
� ",�)
"�
tensor_0��������� 
� �
=__inference_gru_layer_call_and_return_conditional_losses_3120uLJK@�=
6�3
%�"
inputs����������

 
p

 
� ",�)
"�
tensor_0��������� 
� �
=__inference_gru_layer_call_and_return_conditional_losses_3275uLJK@�=
6�3
%�"
inputs����������

 
p 

 
� ",�)
"�
tensor_0��������� 
� �
"__inference_gru_layer_call_fn_2631yLJKO�L
E�B
4�1
/�,
inputs_0������������������

 
p

 
� "!�
unknown��������� �
"__inference_gru_layer_call_fn_2639yLJKO�L
E�B
4�1
/�,
inputs_0������������������

 
p 

 
� "!�
unknown��������� �
"__inference_gru_layer_call_fn_2647jLJK@�=
6�3
%�"
inputs����������

 
p

 
� "!�
unknown��������� �
"__inference_gru_layer_call_fn_2655jLJK@�=
6�3
%�"
inputs����������

 
p 

 
� "!�
unknown��������� �
G__inference_masking_prior_layer_call_and_return_conditional_losses_2560i4�1
*�'
%�"
inputs����������

� "1�.
'�$
tensor_0����������

� �
,__inference_masking_prior_layer_call_fn_2549^4�1
*�'
%�"
inputs����������

� "&�#
unknown����������
�
?__inference_model_layer_call_and_return_conditional_losses_2284xLJKHIA�>
7�4
*�'
input_prior����������

p

 
� ",�)
"�
tensor_0���������
� �
?__inference_model_layer_call_and_return_conditional_losses_2456xLJKHIA�>
7�4
*�'
input_prior����������

p 

 
� ",�)
"�
tensor_0���������
� �
$__inference_model_layer_call_fn_2466mLJKHIA�>
7�4
*�'
input_prior����������

p

 
� "!�
unknown����������
$__inference_model_layer_call_fn_2476mLJKHIA�>
7�4
*�'
input_prior����������

p 

 
� "!�
unknown����������
B__inference_multiply_layer_call_and_return_conditional_losses_2610�d�a
Z�W
U�R
'�$
inputs_0����������

'�$
inputs_1����������

� "1�.
'�$
tensor_0����������

� �
'__inference_multiply_layer_call_fn_2604�d�a
Z�W
U�R
'�$
inputs_0����������

'�$
inputs_1����������

� "&�#
unknown����������
�
"__inference_signature_wrapper_2544�LJKHIH�E
� 
>�;
9
input_prior*�'
input_prior����������
"-�*
(
dense�
dense���������