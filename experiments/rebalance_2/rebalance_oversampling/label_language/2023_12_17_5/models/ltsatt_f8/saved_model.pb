Ч
Т""
D
AddV2
x"T
y"T
z"T"
Ttype:
2	
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
validate_shapebool( 
p
BatchMatMulV2
x"T
y"T
output"T"
Ttype:
2	"
adj_xbool( "
adj_ybool( 

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
resource
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

MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( 
?
Mul
x"T
y"T
z"T"
Ttype:
2	

NoOp
U
NotEqual
x"T
y"T
z
"	
Ttype"$
incompatible_shape_errorbool(
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
dtypetype
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
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
output"out_typeэout_type"	
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

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
С
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
executor_typestring Ј
@
StaticRegexFullMatch	
input

output
"
patternstring
ї
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
А
TensorListFromTensor
tensor"element_dtype
element_shape"
shape_type/
output_handleщшelement_dtype"
element_dtypetype"

shape_typetype:
2	

TensorListReserve
element_shape"
shape_type
num_elements(
handleщшelement_dtype"
element_dtypetype"

shape_typetype:
2	

TensorListStack
input_handle
element_shape
tensor"element_dtype"
element_dtypetype" 
num_elementsintџџџџџџџџџ
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
А
VarHandleOp
resource"
	containerstring "
shared_namestring "

debug_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 

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

&
	ZerosLike
x"T
y"T"	
Ttype"serve*2.15.02v2.15.0-rc1-8-g6887368d6d48СЖ
с
&training_16/Adam/gru_8/gru_cell/bias/vVarHandleOp*
_output_shapes
: *7

debug_name)'training_16/Adam/gru_8/gru_cell/bias/v/*
dtype0*
shape
:`*7
shared_name(&training_16/Adam/gru_8/gru_cell/bias/v
Ё
:training_16/Adam/gru_8/gru_cell/bias/v/Read/ReadVariableOpReadVariableOp&training_16/Adam/gru_8/gru_cell/bias/v*
_output_shapes

:`*
dtype0

2training_16/Adam/gru_8/gru_cell/recurrent_kernel/vVarHandleOp*
_output_shapes
: *C

debug_name53training_16/Adam/gru_8/gru_cell/recurrent_kernel/v/*
dtype0*
shape
: `*C
shared_name42training_16/Adam/gru_8/gru_cell/recurrent_kernel/v
Й
Ftraining_16/Adam/gru_8/gru_cell/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp2training_16/Adam/gru_8/gru_cell/recurrent_kernel/v*
_output_shapes

: `*
dtype0
ч
(training_16/Adam/gru_8/gru_cell/kernel/vVarHandleOp*
_output_shapes
: *9

debug_name+)training_16/Adam/gru_8/gru_cell/kernel/v/*
dtype0*
shape
:`*9
shared_name*(training_16/Adam/gru_8/gru_cell/kernel/v
Ѕ
<training_16/Adam/gru_8/gru_cell/kernel/v/Read/ReadVariableOpReadVariableOp(training_16/Adam/gru_8/gru_cell/kernel/v*
_output_shapes

:`*
dtype0
Ы
 training_16/Adam/dense_44/bias/vVarHandleOp*
_output_shapes
: *1

debug_name#!training_16/Adam/dense_44/bias/v/*
dtype0*
shape:*1
shared_name" training_16/Adam/dense_44/bias/v

4training_16/Adam/dense_44/bias/v/Read/ReadVariableOpReadVariableOp training_16/Adam/dense_44/bias/v*
_output_shapes
:*
dtype0
е
"training_16/Adam/dense_44/kernel/vVarHandleOp*
_output_shapes
: *3

debug_name%#training_16/Adam/dense_44/kernel/v/*
dtype0*
shape
: *3
shared_name$"training_16/Adam/dense_44/kernel/v

6training_16/Adam/dense_44/kernel/v/Read/ReadVariableOpReadVariableOp"training_16/Adam/dense_44/kernel/v*
_output_shapes

: *
dtype0
с
&training_16/Adam/gru_8/gru_cell/bias/mVarHandleOp*
_output_shapes
: *7

debug_name)'training_16/Adam/gru_8/gru_cell/bias/m/*
dtype0*
shape
:`*7
shared_name(&training_16/Adam/gru_8/gru_cell/bias/m
Ё
:training_16/Adam/gru_8/gru_cell/bias/m/Read/ReadVariableOpReadVariableOp&training_16/Adam/gru_8/gru_cell/bias/m*
_output_shapes

:`*
dtype0

2training_16/Adam/gru_8/gru_cell/recurrent_kernel/mVarHandleOp*
_output_shapes
: *C

debug_name53training_16/Adam/gru_8/gru_cell/recurrent_kernel/m/*
dtype0*
shape
: `*C
shared_name42training_16/Adam/gru_8/gru_cell/recurrent_kernel/m
Й
Ftraining_16/Adam/gru_8/gru_cell/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp2training_16/Adam/gru_8/gru_cell/recurrent_kernel/m*
_output_shapes

: `*
dtype0
ч
(training_16/Adam/gru_8/gru_cell/kernel/mVarHandleOp*
_output_shapes
: *9

debug_name+)training_16/Adam/gru_8/gru_cell/kernel/m/*
dtype0*
shape
:`*9
shared_name*(training_16/Adam/gru_8/gru_cell/kernel/m
Ѕ
<training_16/Adam/gru_8/gru_cell/kernel/m/Read/ReadVariableOpReadVariableOp(training_16/Adam/gru_8/gru_cell/kernel/m*
_output_shapes

:`*
dtype0
Ы
 training_16/Adam/dense_44/bias/mVarHandleOp*
_output_shapes
: *1

debug_name#!training_16/Adam/dense_44/bias/m/*
dtype0*
shape:*1
shared_name" training_16/Adam/dense_44/bias/m

4training_16/Adam/dense_44/bias/m/Read/ReadVariableOpReadVariableOp training_16/Adam/dense_44/bias/m*
_output_shapes
:*
dtype0
е
"training_16/Adam/dense_44/kernel/mVarHandleOp*
_output_shapes
: *3

debug_name%#training_16/Adam/dense_44/kernel/m/*
dtype0*
shape
: *3
shared_name$"training_16/Adam/dense_44/kernel/m

6training_16/Adam/dense_44/kernel/m/Read/ReadVariableOpReadVariableOp"training_16/Adam/dense_44/kernel/m*
_output_shapes

: *
dtype0

false_negatives_8VarHandleOp*
_output_shapes
: *"

debug_namefalse_negatives_8/*
dtype0*
shape:Ш*"
shared_namefalse_negatives_8
t
%false_negatives_8/Read/ReadVariableOpReadVariableOpfalse_negatives_8*
_output_shapes	
:Ш*
dtype0

false_positives_8VarHandleOp*
_output_shapes
: *"

debug_namefalse_positives_8/*
dtype0*
shape:Ш*"
shared_namefalse_positives_8
t
%false_positives_8/Read/ReadVariableOpReadVariableOpfalse_positives_8*
_output_shapes	
:Ш*
dtype0

true_negatives_8VarHandleOp*
_output_shapes
: *!

debug_nametrue_negatives_8/*
dtype0*
shape:Ш*!
shared_nametrue_negatives_8
r
$true_negatives_8/Read/ReadVariableOpReadVariableOptrue_negatives_8*
_output_shapes	
:Ш*
dtype0

true_positives_8VarHandleOp*
_output_shapes
: *!

debug_nametrue_positives_8/*
dtype0*
shape:Ш*!
shared_nametrue_positives_8
r
$true_positives_8/Read/ReadVariableOpReadVariableOptrue_positives_8*
_output_shapes	
:Ш*
dtype0
|
count_8VarHandleOp*
_output_shapes
: *

debug_name
count_8/*
dtype0*
shape: *
shared_name	count_8
[
count_8/Read/ReadVariableOpReadVariableOpcount_8*
_output_shapes
: *
dtype0
|
total_8VarHandleOp*
_output_shapes
: *

debug_name
total_8/*
dtype0*
shape: *
shared_name	total_8
[
total_8/Read/ReadVariableOpReadVariableOptotal_8*
_output_shapes
: *
dtype0
С
training_16/Adam/learning_rateVarHandleOp*
_output_shapes
: */

debug_name!training_16/Adam/learning_rate/*
dtype0*
shape: */
shared_name training_16/Adam/learning_rate

2training_16/Adam/learning_rate/Read/ReadVariableOpReadVariableOptraining_16/Adam/learning_rate*
_output_shapes
: *
dtype0
Љ
training_16/Adam/decayVarHandleOp*
_output_shapes
: *'

debug_nametraining_16/Adam/decay/*
dtype0*
shape: *'
shared_nametraining_16/Adam/decay
y
*training_16/Adam/decay/Read/ReadVariableOpReadVariableOptraining_16/Adam/decay*
_output_shapes
: *
dtype0
Ќ
training_16/Adam/beta_2VarHandleOp*
_output_shapes
: *(

debug_nametraining_16/Adam/beta_2/*
dtype0*
shape: *(
shared_nametraining_16/Adam/beta_2
{
+training_16/Adam/beta_2/Read/ReadVariableOpReadVariableOptraining_16/Adam/beta_2*
_output_shapes
: *
dtype0
Ќ
training_16/Adam/beta_1VarHandleOp*
_output_shapes
: *(

debug_nametraining_16/Adam/beta_1/*
dtype0*
shape: *(
shared_nametraining_16/Adam/beta_1
{
+training_16/Adam/beta_1/Read/ReadVariableOpReadVariableOptraining_16/Adam/beta_1*
_output_shapes
: *
dtype0
І
training_16/Adam/iterVarHandleOp*
_output_shapes
: *&

debug_nametraining_16/Adam/iter/*
dtype0	*
shape: *&
shared_nametraining_16/Adam/iter
w
)training_16/Adam/iter/Read/ReadVariableOpReadVariableOptraining_16/Adam/iter*
_output_shapes
: *
dtype0	
Ј
gru_8/gru_cell/biasVarHandleOp*
_output_shapes
: *$

debug_namegru_8/gru_cell/bias/*
dtype0*
shape
:`*$
shared_namegru_8/gru_cell/bias
{
'gru_8/gru_cell/bias/Read/ReadVariableOpReadVariableOpgru_8/gru_cell/bias*
_output_shapes

:`*
dtype0
Ь
gru_8/gru_cell/recurrent_kernelVarHandleOp*
_output_shapes
: *0

debug_name" gru_8/gru_cell/recurrent_kernel/*
dtype0*
shape
: `*0
shared_name!gru_8/gru_cell/recurrent_kernel

3gru_8/gru_cell/recurrent_kernel/Read/ReadVariableOpReadVariableOpgru_8/gru_cell/recurrent_kernel*
_output_shapes

: `*
dtype0
Ў
gru_8/gru_cell/kernelVarHandleOp*
_output_shapes
: *&

debug_namegru_8/gru_cell/kernel/*
dtype0*
shape
:`*&
shared_namegru_8/gru_cell/kernel

)gru_8/gru_cell/kernel/Read/ReadVariableOpReadVariableOpgru_8/gru_cell/kernel*
_output_shapes

:`*
dtype0

dense_44/biasVarHandleOp*
_output_shapes
: *

debug_namedense_44/bias/*
dtype0*
shape:*
shared_namedense_44/bias
k
!dense_44/bias/Read/ReadVariableOpReadVariableOpdense_44/bias*
_output_shapes
:*
dtype0

dense_44/kernelVarHandleOp*
_output_shapes
: * 

debug_namedense_44/kernel/*
dtype0*
shape
: * 
shared_namedense_44/kernel
s
#dense_44/kernel/Read/ReadVariableOpReadVariableOpdense_44/kernel*
_output_shapes

: *
dtype0

serving_default_input_priorPlaceholder*,
_output_shapes
:џџџџџџџџџГ
*
dtype0*!
shape:џџџџџџџџџГ

Ќ
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_priorgru_8/gru_cell/biasgru_8/gru_cell/kernelgru_8/gru_cell/recurrent_kerneldense_44/kerneldense_44/bias*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8 *,
f'R%
#__inference_signature_wrapper_49807

NoOpNoOp
J
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*НI
valueГIBАI BЉI
ѕ
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

	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses* 

	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
axes* 

 	variables
!trainable_variables
"regularization_losses
#	keras_api
$__call__
*%&call_and_return_all_conditional_losses* 

&	variables
'trainable_variables
(regularization_losses
)	keras_api
*__call__
*+&call_and_return_all_conditional_losses
,axes* 

-	variables
.trainable_variables
/regularization_losses
0	keras_api
1__call__
*2&call_and_return_all_conditional_losses* 

3	variables
4trainable_variables
5regularization_losses
6	keras_api
7__call__
*8&call_and_return_all_conditional_losses* 
С
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
І
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
А
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
Ј
Viter

Wbeta_1

Xbeta_2
	Ydecay
Zlearning_rateHmИImЙJmКKmЛLmМHvНIvОJvПKvРLvС*

[serving_default* 
* 
* 
* 

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

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

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

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

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

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
3	variables
4trainable_variables
5regularization_losses
7__call__
*8&call_and_return_all_conditional_losses
&8"call_and_return_conditional_losses* 

trace_0* 

trace_0* 

J0
K1
L2*

J0
K1
L2*
* 
Ѕ
states
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
9	variables
:trainable_variables
;regularization_losses
=__call__
*>&call_and_return_all_conditional_losses
&>"call_and_return_conditional_losses*
:
trace_0
trace_1
trace_2
trace_3* 
:
trace_0
trace_1
trace_2
trace_3* 
* 
к
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
_random_generator

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

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
B	variables
Ctrainable_variables
Dregularization_losses
F__call__
*G&call_and_return_all_conditional_losses
&G"call_and_return_conditional_losses*

 trace_0* 

Ёtrace_0* 
_Y
VARIABLE_VALUEdense_44/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_44/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEgru_8/gru_cell/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEgru_8/gru_cell/recurrent_kernel&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEgru_8/gru_cell/bias&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
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
Ђ0
Ѓ1*
* 
* 
* 
* 
* 
* 
XR
VARIABLE_VALUEtraining_16/Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEtraining_16/Adam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEtraining_16/Adam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEtraining_16/Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEtraining_16/Adam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
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

Єnon_trainable_variables
Ѕlayers
Іmetrics
 Їlayer_regularization_losses
Јlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*

Љtrace_0
Њtrace_1* 

Ћtrace_0
Ќtrace_1* 
* 
* 
* 
* 
* 
* 
* 
* 
M
­	variables
Ў	keras_api

Џtotal

Аcount
Б
_fn_kwargs*
z
В	variables
Г	keras_api
Дtrue_positives
Еtrue_negatives
Жfalse_positives
Зfalse_negatives*
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
Џ0
А1*

­	variables*
UO
VARIABLE_VALUEtotal_84keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_84keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*
* 
$
Д0
Е1
Ж2
З3*

В	variables*
ga
VARIABLE_VALUEtrue_positives_8=keras_api/metrics/1/true_positives/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUEtrue_negatives_8=keras_api/metrics/1/true_negatives/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEfalse_positives_8>keras_api/metrics/1/false_positives/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEfalse_negatives_8>keras_api/metrics/1/false_negatives/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE"training_16/Adam/dense_44/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE training_16/Adam/dense_44/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUE(training_16/Adam/gru_8/gru_cell/kernel/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE2training_16/Adam/gru_8/gru_cell/recurrent_kernel/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUE&training_16/Adam/gru_8/gru_cell/bias/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE"training_16/Adam/dense_44/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE training_16/Adam/dense_44/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUE(training_16/Adam/gru_8/gru_cell/kernel/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE2training_16/Adam/gru_8/gru_cell/recurrent_kernel/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUE&training_16/Adam/gru_8/gru_cell/bias/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenamedense_44/kerneldense_44/biasgru_8/gru_cell/kernelgru_8/gru_cell/recurrent_kernelgru_8/gru_cell/biastraining_16/Adam/itertraining_16/Adam/beta_1training_16/Adam/beta_2training_16/Adam/decaytraining_16/Adam/learning_ratetotal_8count_8true_positives_8true_negatives_8false_positives_8false_negatives_8"training_16/Adam/dense_44/kernel/m training_16/Adam/dense_44/bias/m(training_16/Adam/gru_8/gru_cell/kernel/m2training_16/Adam/gru_8/gru_cell/recurrent_kernel/m&training_16/Adam/gru_8/gru_cell/bias/m"training_16/Adam/dense_44/kernel/v training_16/Adam/dense_44/bias/v(training_16/Adam/gru_8/gru_cell/kernel/v2training_16/Adam/gru_8/gru_cell/recurrent_kernel/v&training_16/Adam/gru_8/gru_cell/bias/vConst*'
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
GPU 2J 8 *'
f"R 
__inference__traced_save_50834

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_44/kerneldense_44/biasgru_8/gru_cell/kernelgru_8/gru_cell/recurrent_kernelgru_8/gru_cell/biastraining_16/Adam/itertraining_16/Adam/beta_1training_16/Adam/beta_2training_16/Adam/decaytraining_16/Adam/learning_ratetotal_8count_8true_positives_8true_negatives_8false_positives_8false_negatives_8"training_16/Adam/dense_44/kernel/m training_16/Adam/dense_44/bias/m(training_16/Adam/gru_8/gru_cell/kernel/m2training_16/Adam/gru_8/gru_cell/recurrent_kernel/m&training_16/Adam/gru_8/gru_cell/bias/m"training_16/Adam/dense_44/kernel/v training_16/Adam/dense_44/bias/v(training_16/Adam/gru_8/gru_cell/kernel/v2training_16/Adam/gru_8/gru_cell/recurrent_kernel/v&training_16/Adam/gru_8/gru_cell/bias/v*&
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
GPU 2J 8 **
f%R#
!__inference__traced_restore_50921Б
Ъ
d
H__inference_masking_prior_layer_call_and_return_conditional_losses_49823

inputs
identityO

NotEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  Пh
NotEqualNotEqualinputsNotEqual/y:output:0*
T0*,
_output_shapes
:џџџџџџџџџГ
`
Any/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџw
AnyAnyNotEqual:z:0Any/reduction_indices:output:0*,
_output_shapes
:џџџџџџџџџГ*
	keep_dims(`
CastCastAny:output:0*

DstT0*

SrcT0
*,
_output_shapes
:џџџџџџџџџГS
mulMulinputsCast:y:0*
T0*,
_output_shapes
:џџџџџџџџџГ
s
SqueezeSqueezeAny:output:0*
T0
*(
_output_shapes
:џџџџџџџџџГ*
squeeze_dims

џџџџџџџџџT
IdentityIdentitymul:z:0*
T0*,
_output_shapes
:џџџџџџџџџГ
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџГ
:T P
,
_output_shapes
:џџџџџџџџџГ

 
_user_specified_nameinputs
М
R
&__inference_dot_17_layer_call_fn_49854
inputs_0
inputs_1
identityО
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџГ
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dot_17_layer_call_and_return_conditional_losses_49356e
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:џџџџџџџџџГ
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:џџџџџџџџџГГ:џџџџџџџџџГ
:VR
,
_output_shapes
:џџџџџџџџџГ

"
_user_specified_name
inputs_1:W S
-
_output_shapes
:џџџџџџџџџГГ
"
_user_specified_name
inputs_0
й	
с
%__inference_gru_8_layer_call_fn_49918

inputs%
gru_8_gru_cell_bias:`'
gru_8_gru_cell_kernel:`1
gru_8_gru_cell_recurrent_kernel: `
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsgru_8_gru_cell_biasgru_8_gru_cell_kernelgru_8_gru_cell_recurrent_kernel*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_gru_8_layer_call_and_return_conditional_losses_49710o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:џџџџџџџџџГ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:?;
9
_user_specified_name!gru_8/gru_cell/recurrent_kernel:51
/
_user_specified_namegru_8/gru_cell/kernel:3/
-
_user_specified_namegru_8/gru_cell/bias:T P
,
_output_shapes
:џџџџџџџџџГ
 
_user_specified_nameinputs
Ш
Y
-__inference_concatenate_8_layer_call_fn_49879
inputs_0
inputs_1
identityХ
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџГ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_concatenate_8_layer_call_and_return_conditional_losses_49371e
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:џџџџџџџџџГ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:џџџџџџџџџГ
:џџџџџџџџџГ
:VR
,
_output_shapes
:џџџџџџџџџГ

"
_user_specified_name
inputs_1:V R
,
_output_shapes
:џџџџџџџџџГ

"
_user_specified_name
inputs_0
вN
Ђ
@__inference_gru_8_layer_call_and_return_conditional_losses_50228
inputs_0=
+gru_cell_readvariableop_gru_8_gru_cell_bias:`F
4gru_cell_matmul_readvariableop_gru_8_gru_cell_kernel:`R
@gru_cell_matmul_1_readvariableop_gru_8_gru_cell_recurrent_kernel: `
identityЂgru_cell/MatMul/ReadVariableOpЂ gru_cell/MatMul_1/ReadVariableOpЂgru_cell/ReadVariableOpЂwhileK
ShapeShapeinputs_0*
T0*
_output_shapes
::эЯ]
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
valueB:б
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
:џџџџџџџџџ c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџR
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
::эЯ_
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
valueB:л
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
џџџџџџџџџД
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   р
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв_
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
valueB:щ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask
gru_cell/ReadVariableOpReadVariableOp+gru_cell_readvariableop_gru_8_gru_cell_bias*
_output_shapes

:`*
dtype0q
gru_cell/unstackUnpackgru_cell/ReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
num
gru_cell/MatMul/ReadVariableOpReadVariableOp4gru_cell_matmul_readvariableop_gru_8_gru_cell_kernel*
_output_shapes

:`*
dtype0
gru_cell/MatMulMatMulstrided_slice_2:output:0&gru_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ`
gru_cell/BiasAddBiasAddgru_cell/MatMul:product:0gru_cell/unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџ`c
gru_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџО
gru_cell/splitSplit!gru_cell/split/split_dim:output:0gru_cell/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ *
	num_splitЁ
 gru_cell/MatMul_1/ReadVariableOpReadVariableOp@gru_cell_matmul_1_readvariableop_gru_8_gru_cell_recurrent_kernel*
_output_shapes

: `*
dtype0
gru_cell/MatMul_1MatMulzeros:output:0(gru_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ`
gru_cell/BiasAdd_1BiasAddgru_cell/MatMul_1:product:0gru_cell/unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџ`c
gru_cell/ConstConst*
_output_shapes
:*
dtype0*!
valueB"        џџџџe
gru_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџъ
gru_cell/split_1SplitVgru_cell/BiasAdd_1:output:0gru_cell/Const:output:0#gru_cell/split_1/split_dim:output:0*

Tlen0*
T0*M
_output_shapes;
9:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ *
	num_split{
gru_cell/addAddV2gru_cell/split:output:0gru_cell/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ _
gru_cell/SigmoidSigmoidgru_cell/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ }
gru_cell/add_1AddV2gru_cell/split:output:1gru_cell/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ c
gru_cell/Sigmoid_1Sigmoidgru_cell/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ x
gru_cell/mulMulgru_cell/Sigmoid_1:y:0gru_cell/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ t
gru_cell/add_2AddV2gru_cell/split:output:2gru_cell/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ [
gru_cell/TanhTanhgru_cell/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ m
gru_cell/mul_1Mulgru_cell/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:џџџџџџџџџ S
gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?t
gru_cell/subSubgru_cell/sub/x:output:0gru_cell/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ l
gru_cell/mul_2Mulgru_cell/sub:z:0gru_cell/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ q
gru_cell/add_3AddV2gru_cell/mul_1:z:0gru_cell/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :Х
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшвF
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
џџџџџџџџџT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Х
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+gru_cell_readvariableop_gru_8_gru_cell_bias4gru_cell_matmul_readvariableop_gru_8_gru_cell_kernel@gru_cell_matmul_1_readvariableop_gru_8_gru_cell_recurrent_kernel*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :џџџџџџџџџ : : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_50138*
condR
while_cond_50137*8
output_shapes'
%: : : : :џџџџџџџџџ : : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    ж
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:џџџџџџџџџ *
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ 
NoOpNoOp^gru_cell/MatMul/ReadVariableOp!^gru_cell/MatMul_1/ReadVariableOp^gru_cell/ReadVariableOp^while*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:џџџџџџџџџџџџџџџџџџ: : : 2@
gru_cell/MatMul/ReadVariableOpgru_cell/MatMul/ReadVariableOp2D
 gru_cell/MatMul_1/ReadVariableOp gru_cell/MatMul_1/ReadVariableOp22
gru_cell/ReadVariableOpgru_cell/ReadVariableOp2
whilewhile:?;
9
_user_specified_name!gru_8/gru_cell/recurrent_kernel:51
/
_user_specified_namegru_8/gru_cell/kernel:3/
-
_user_specified_namegru_8/gru_cell/bias:^ Z
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs_0
і
у
while_cond_49619
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
less_strided_slice_1-
)while_cond_49619___redundant_placeholder0-
)while_cond_49619___redundant_placeholder1-
)while_cond_49619___redundant_placeholder2-
)while_cond_49619___redundant_placeholder3
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
-: : : : :џџџџџџџџџ : :::::
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
:џџџџџџџџџ :
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
Б!

while_body_49219
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
strided_slice_1_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_00
gru_cell_gru_8_gru_cell_bias_0:`2
 gru_cell_gru_8_gru_cell_kernel_0:`<
*gru_cell_gru_8_gru_cell_recurrent_kernel_0: `
identity

identity_1

identity_2

identity_3

identity_4
strided_slice_1O
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor.
gru_cell_gru_8_gru_cell_bias:`0
gru_cell_gru_8_gru_cell_kernel:`:
(gru_cell_gru_8_gru_cell_recurrent_kernel: `Ђ gru_cell/StatefulPartitionedCall
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype0
 gru_cell/StatefulPartitionedCallStatefulPartitionedCall*TensorArrayV2Read/TensorListGetItem:item:0placeholder_2gru_cell_gru_8_gru_cell_bias_0 gru_cell_gru_8_gru_cell_kernel_0*gru_cell_gru_8_gru_cell_recurrent_kernel_0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:џџџџџџџџџ :џџџџџџџџџ *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_gru_cell_layer_call_and_return_conditional_losses_49208l
*TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : ш
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_13TensorArrayV2Write/TensorListSetItem/index:output:0)gru_cell/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:щшвG
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
:џџџџџџџџџ E
NoOpNoOp!^gru_cell/StatefulPartitionedCall*
_output_shapes
 ">
gru_cell_gru_8_gru_cell_biasgru_cell_gru_8_gru_cell_bias_0"B
gru_cell_gru_8_gru_cell_kernel gru_cell_gru_8_gru_cell_kernel_0"V
(gru_cell_gru_8_gru_cell_recurrent_kernel*gru_cell_gru_8_gru_cell_recurrent_kernel_0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
identityIdentity:output:0"$
strided_slice_1strided_slice_1_0"
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :џџџџџџџџџ : : : : : 2D
 gru_cell/StatefulPartitionedCall gru_cell/StatefulPartitionedCall:?	;
9
_user_specified_name!gru_8/gru_cell/recurrent_kernel:51
/
_user_specified_namegru_8/gru_cell/kernel:3/
-
_user_specified_namegru_8/gru_cell/bias:_[
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
:џџџџџџџџџ :
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
й9
К
while_body_50138
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
strided_slice_1_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0?
-gru_cell_readvariableop_gru_8_gru_cell_bias_0:`H
6gru_cell_matmul_readvariableop_gru_8_gru_cell_kernel_0:`T
Bgru_cell_matmul_1_readvariableop_gru_8_gru_cell_recurrent_kernel_0: `
identity

identity_1

identity_2

identity_3

identity_4
strided_slice_1O
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor=
+gru_cell_readvariableop_gru_8_gru_cell_bias:`F
4gru_cell_matmul_readvariableop_gru_8_gru_cell_kernel:`R
@gru_cell_matmul_1_readvariableop_gru_8_gru_cell_recurrent_kernel: `Ђgru_cell/MatMul/ReadVariableOpЂ gru_cell/MatMul_1/ReadVariableOpЂgru_cell/ReadVariableOp
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype0
gru_cell/ReadVariableOpReadVariableOp-gru_cell_readvariableop_gru_8_gru_cell_bias_0*
_output_shapes

:`*
dtype0q
gru_cell/unstackUnpackgru_cell/ReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
num
gru_cell/MatMul/ReadVariableOpReadVariableOp6gru_cell_matmul_readvariableop_gru_8_gru_cell_kernel_0*
_output_shapes

:`*
dtype0
gru_cell/MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0&gru_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ`
gru_cell/BiasAddBiasAddgru_cell/MatMul:product:0gru_cell/unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџ`c
gru_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџО
gru_cell/splitSplit!gru_cell/split/split_dim:output:0gru_cell/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ *
	num_splitЃ
 gru_cell/MatMul_1/ReadVariableOpReadVariableOpBgru_cell_matmul_1_readvariableop_gru_8_gru_cell_recurrent_kernel_0*
_output_shapes

: `*
dtype0
gru_cell/MatMul_1MatMulplaceholder_2(gru_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ`
gru_cell/BiasAdd_1BiasAddgru_cell/MatMul_1:product:0gru_cell/unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџ`c
gru_cell/ConstConst*
_output_shapes
:*
dtype0*!
valueB"        џџџџe
gru_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџъ
gru_cell/split_1SplitVgru_cell/BiasAdd_1:output:0gru_cell/Const:output:0#gru_cell/split_1/split_dim:output:0*

Tlen0*
T0*M
_output_shapes;
9:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ *
	num_split{
gru_cell/addAddV2gru_cell/split:output:0gru_cell/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ _
gru_cell/SigmoidSigmoidgru_cell/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ }
gru_cell/add_1AddV2gru_cell/split:output:1gru_cell/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ c
gru_cell/Sigmoid_1Sigmoidgru_cell/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ x
gru_cell/mulMulgru_cell/Sigmoid_1:y:0gru_cell/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ t
gru_cell/add_2AddV2gru_cell/split:output:2gru_cell/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ [
gru_cell/TanhTanhgru_cell/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ l
gru_cell/mul_1Mulgru_cell/Sigmoid:y:0placeholder_2*
T0*'
_output_shapes
:џџџџџџџџџ S
gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?t
gru_cell/subSubgru_cell/sub/x:output:0gru_cell/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ l
gru_cell/mul_2Mulgru_cell/sub:z:0gru_cell/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ q
gru_cell/add_3AddV2gru_cell/mul_1:z:0gru_cell/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ l
*TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : б
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_13TensorArrayV2Write/TensorListSetItem/index:output:0gru_cell/add_3:z:0*
_output_shapes
: *
element_dtype0:щшвG
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
:џџџџџџџџџ 
NoOpNoOp^gru_cell/MatMul/ReadVariableOp!^gru_cell/MatMul_1/ReadVariableOp^gru_cell/ReadVariableOp*
_output_shapes
 "
@gru_cell_matmul_1_readvariableop_gru_8_gru_cell_recurrent_kernelBgru_cell_matmul_1_readvariableop_gru_8_gru_cell_recurrent_kernel_0"n
4gru_cell_matmul_readvariableop_gru_8_gru_cell_kernel6gru_cell_matmul_readvariableop_gru_8_gru_cell_kernel_0"\
+gru_cell_readvariableop_gru_8_gru_cell_bias-gru_cell_readvariableop_gru_8_gru_cell_bias_0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
identityIdentity:output:0"$
strided_slice_1strided_slice_1_0"
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :џџџџџџџџџ : : : : : 2@
gru_cell/MatMul/ReadVariableOpgru_cell/MatMul/ReadVariableOp2D
 gru_cell/MatMul_1/ReadVariableOp gru_cell/MatMul_1/ReadVariableOp22
gru_cell/ReadVariableOpgru_cell/ReadVariableOp:?	;
9
_user_specified_name!gru_8/gru_cell/recurrent_kernel:51
/
_user_specified_namegru_8/gru_cell/kernel:3/
-
_user_specified_namegru_8/gru_cell/bias:_[
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
:џџџџџџџџџ :
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
й9
К
while_body_50448
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
strided_slice_1_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0?
-gru_cell_readvariableop_gru_8_gru_cell_bias_0:`H
6gru_cell_matmul_readvariableop_gru_8_gru_cell_kernel_0:`T
Bgru_cell_matmul_1_readvariableop_gru_8_gru_cell_recurrent_kernel_0: `
identity

identity_1

identity_2

identity_3

identity_4
strided_slice_1O
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor=
+gru_cell_readvariableop_gru_8_gru_cell_bias:`F
4gru_cell_matmul_readvariableop_gru_8_gru_cell_kernel:`R
@gru_cell_matmul_1_readvariableop_gru_8_gru_cell_recurrent_kernel: `Ђgru_cell/MatMul/ReadVariableOpЂ gru_cell/MatMul_1/ReadVariableOpЂgru_cell/ReadVariableOp
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype0
gru_cell/ReadVariableOpReadVariableOp-gru_cell_readvariableop_gru_8_gru_cell_bias_0*
_output_shapes

:`*
dtype0q
gru_cell/unstackUnpackgru_cell/ReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
num
gru_cell/MatMul/ReadVariableOpReadVariableOp6gru_cell_matmul_readvariableop_gru_8_gru_cell_kernel_0*
_output_shapes

:`*
dtype0
gru_cell/MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0&gru_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ`
gru_cell/BiasAddBiasAddgru_cell/MatMul:product:0gru_cell/unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџ`c
gru_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџО
gru_cell/splitSplit!gru_cell/split/split_dim:output:0gru_cell/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ *
	num_splitЃ
 gru_cell/MatMul_1/ReadVariableOpReadVariableOpBgru_cell_matmul_1_readvariableop_gru_8_gru_cell_recurrent_kernel_0*
_output_shapes

: `*
dtype0
gru_cell/MatMul_1MatMulplaceholder_2(gru_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ`
gru_cell/BiasAdd_1BiasAddgru_cell/MatMul_1:product:0gru_cell/unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџ`c
gru_cell/ConstConst*
_output_shapes
:*
dtype0*!
valueB"        џџџџe
gru_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџъ
gru_cell/split_1SplitVgru_cell/BiasAdd_1:output:0gru_cell/Const:output:0#gru_cell/split_1/split_dim:output:0*

Tlen0*
T0*M
_output_shapes;
9:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ *
	num_split{
gru_cell/addAddV2gru_cell/split:output:0gru_cell/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ _
gru_cell/SigmoidSigmoidgru_cell/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ }
gru_cell/add_1AddV2gru_cell/split:output:1gru_cell/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ c
gru_cell/Sigmoid_1Sigmoidgru_cell/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ x
gru_cell/mulMulgru_cell/Sigmoid_1:y:0gru_cell/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ t
gru_cell/add_2AddV2gru_cell/split:output:2gru_cell/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ [
gru_cell/TanhTanhgru_cell/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ l
gru_cell/mul_1Mulgru_cell/Sigmoid:y:0placeholder_2*
T0*'
_output_shapes
:џџџџџџџџџ S
gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?t
gru_cell/subSubgru_cell/sub/x:output:0gru_cell/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ l
gru_cell/mul_2Mulgru_cell/sub:z:0gru_cell/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ q
gru_cell/add_3AddV2gru_cell/mul_1:z:0gru_cell/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ l
*TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : б
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_13TensorArrayV2Write/TensorListSetItem/index:output:0gru_cell/add_3:z:0*
_output_shapes
: *
element_dtype0:щшвG
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
:џџџџџџџџџ 
NoOpNoOp^gru_cell/MatMul/ReadVariableOp!^gru_cell/MatMul_1/ReadVariableOp^gru_cell/ReadVariableOp*
_output_shapes
 "
@gru_cell_matmul_1_readvariableop_gru_8_gru_cell_recurrent_kernelBgru_cell_matmul_1_readvariableop_gru_8_gru_cell_recurrent_kernel_0"n
4gru_cell_matmul_readvariableop_gru_8_gru_cell_kernel6gru_cell_matmul_readvariableop_gru_8_gru_cell_kernel_0"\
+gru_cell_readvariableop_gru_8_gru_cell_bias-gru_cell_readvariableop_gru_8_gru_cell_bias_0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
identityIdentity:output:0"$
strided_slice_1strided_slice_1_0"
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :џџџџџџџџџ : : : : : 2@
gru_cell/MatMul/ReadVariableOpgru_cell/MatMul/ReadVariableOp2D
 gru_cell/MatMul_1/ReadVariableOp gru_cell/MatMul_1/ReadVariableOp22
gru_cell/ReadVariableOpgru_cell/ReadVariableOp:?	;
9
_user_specified_name!gru_8/gru_cell/recurrent_kernel:51
/
_user_specified_namegru_8/gru_cell/kernel:3/
-
_user_specified_namegru_8/gru_cell/bias:_[
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
:џџџџџџџџџ :
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
Х
m
A__inference_dot_17_layer_call_and_return_conditional_losses_49861
inputs_0
inputs_1
identityb
MatMulBatchMatMulV2inputs_0inputs_1*
T0*,
_output_shapes
:џџџџџџџџџГ
R
ShapeShapeMatMul:output:0*
T0*
_output_shapes
::эЯ\
IdentityIdentityMatMul:output:0*
T0*,
_output_shapes
:џџџџџџџџџГ
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:џџџџџџџџџГГ:џџџџџџџџџГ
:VR
,
_output_shapes
:џџџџџџџџџГ

"
_user_specified_name
inputs_1:W S
-
_output_shapes
:џџџџџџџџџГГ
"
_user_specified_name
inputs_0
Й
I
-__inference_masking_prior_layer_call_fn_49812

inputs
identityИ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџГ
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_masking_prior_layer_call_and_return_conditional_losses_49332e
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:џџџџџџџџџГ
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџГ
:T P
,
_output_shapes
:џџџџџџџџџГ

 
_user_specified_nameinputs
вN
Ђ
@__inference_gru_8_layer_call_and_return_conditional_losses_50073
inputs_0=
+gru_cell_readvariableop_gru_8_gru_cell_bias:`F
4gru_cell_matmul_readvariableop_gru_8_gru_cell_kernel:`R
@gru_cell_matmul_1_readvariableop_gru_8_gru_cell_recurrent_kernel: `
identityЂgru_cell/MatMul/ReadVariableOpЂ gru_cell/MatMul_1/ReadVariableOpЂgru_cell/ReadVariableOpЂwhileK
ShapeShapeinputs_0*
T0*
_output_shapes
::эЯ]
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
valueB:б
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
:џџџџџџџџџ c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџR
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
::эЯ_
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
valueB:л
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
џџџџџџџџџД
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   р
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв_
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
valueB:щ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask
gru_cell/ReadVariableOpReadVariableOp+gru_cell_readvariableop_gru_8_gru_cell_bias*
_output_shapes

:`*
dtype0q
gru_cell/unstackUnpackgru_cell/ReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
num
gru_cell/MatMul/ReadVariableOpReadVariableOp4gru_cell_matmul_readvariableop_gru_8_gru_cell_kernel*
_output_shapes

:`*
dtype0
gru_cell/MatMulMatMulstrided_slice_2:output:0&gru_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ`
gru_cell/BiasAddBiasAddgru_cell/MatMul:product:0gru_cell/unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџ`c
gru_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџО
gru_cell/splitSplit!gru_cell/split/split_dim:output:0gru_cell/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ *
	num_splitЁ
 gru_cell/MatMul_1/ReadVariableOpReadVariableOp@gru_cell_matmul_1_readvariableop_gru_8_gru_cell_recurrent_kernel*
_output_shapes

: `*
dtype0
gru_cell/MatMul_1MatMulzeros:output:0(gru_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ`
gru_cell/BiasAdd_1BiasAddgru_cell/MatMul_1:product:0gru_cell/unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџ`c
gru_cell/ConstConst*
_output_shapes
:*
dtype0*!
valueB"        џџџџe
gru_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџъ
gru_cell/split_1SplitVgru_cell/BiasAdd_1:output:0gru_cell/Const:output:0#gru_cell/split_1/split_dim:output:0*

Tlen0*
T0*M
_output_shapes;
9:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ *
	num_split{
gru_cell/addAddV2gru_cell/split:output:0gru_cell/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ _
gru_cell/SigmoidSigmoidgru_cell/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ }
gru_cell/add_1AddV2gru_cell/split:output:1gru_cell/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ c
gru_cell/Sigmoid_1Sigmoidgru_cell/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ x
gru_cell/mulMulgru_cell/Sigmoid_1:y:0gru_cell/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ t
gru_cell/add_2AddV2gru_cell/split:output:2gru_cell/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ [
gru_cell/TanhTanhgru_cell/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ m
gru_cell/mul_1Mulgru_cell/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:џџџџџџџџџ S
gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?t
gru_cell/subSubgru_cell/sub/x:output:0gru_cell/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ l
gru_cell/mul_2Mulgru_cell/sub:z:0gru_cell/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ q
gru_cell/add_3AddV2gru_cell/mul_1:z:0gru_cell/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :Х
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшвF
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
џџџџџџџџџT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Х
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+gru_cell_readvariableop_gru_8_gru_cell_bias4gru_cell_matmul_readvariableop_gru_8_gru_cell_kernel@gru_cell_matmul_1_readvariableop_gru_8_gru_cell_recurrent_kernel*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :џџџџџџџџџ : : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_49983*
condR
while_cond_49982*8
output_shapes'
%: : : : :џџџџџџџџџ : : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    ж
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:џџџџџџџџџ *
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ 
NoOpNoOp^gru_cell/MatMul/ReadVariableOp!^gru_cell/MatMul_1/ReadVariableOp^gru_cell/ReadVariableOp^while*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:џџџџџџџџџџџџџџџџџџ: : : 2@
gru_cell/MatMul/ReadVariableOpgru_cell/MatMul/ReadVariableOp2D
 gru_cell/MatMul_1/ReadVariableOp gru_cell/MatMul_1/ReadVariableOp22
gru_cell/ReadVariableOpgru_cell/ReadVariableOp2
whilewhile:?;
9
_user_specified_name!gru_8/gru_cell/recurrent_kernel:51
/
_user_specified_namegru_8/gru_cell/kernel:3/
-
_user_specified_namegru_8/gru_cell/bias:^ Z
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs_0
ВN
 
@__inference_gru_8_layer_call_and_return_conditional_losses_50383

inputs=
+gru_cell_readvariableop_gru_8_gru_cell_bias:`F
4gru_cell_matmul_readvariableop_gru_8_gru_cell_kernel:`R
@gru_cell_matmul_1_readvariableop_gru_8_gru_cell_recurrent_kernel: `
identityЂgru_cell/MatMul/ReadVariableOpЂ gru_cell/MatMul_1/ReadVariableOpЂgru_cell/ReadVariableOpЂwhileI
ShapeShapeinputs*
T0*
_output_shapes
::эЯ]
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
valueB:б
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
:џџџџџџџџџ c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          n
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:ГџџџџџџџџџR
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
::эЯ_
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
valueB:л
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
џџџџџџџџџД
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   р
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв_
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
valueB:щ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask
gru_cell/ReadVariableOpReadVariableOp+gru_cell_readvariableop_gru_8_gru_cell_bias*
_output_shapes

:`*
dtype0q
gru_cell/unstackUnpackgru_cell/ReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
num
gru_cell/MatMul/ReadVariableOpReadVariableOp4gru_cell_matmul_readvariableop_gru_8_gru_cell_kernel*
_output_shapes

:`*
dtype0
gru_cell/MatMulMatMulstrided_slice_2:output:0&gru_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ`
gru_cell/BiasAddBiasAddgru_cell/MatMul:product:0gru_cell/unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџ`c
gru_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџО
gru_cell/splitSplit!gru_cell/split/split_dim:output:0gru_cell/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ *
	num_splitЁ
 gru_cell/MatMul_1/ReadVariableOpReadVariableOp@gru_cell_matmul_1_readvariableop_gru_8_gru_cell_recurrent_kernel*
_output_shapes

: `*
dtype0
gru_cell/MatMul_1MatMulzeros:output:0(gru_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ`
gru_cell/BiasAdd_1BiasAddgru_cell/MatMul_1:product:0gru_cell/unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџ`c
gru_cell/ConstConst*
_output_shapes
:*
dtype0*!
valueB"        џџџџe
gru_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџъ
gru_cell/split_1SplitVgru_cell/BiasAdd_1:output:0gru_cell/Const:output:0#gru_cell/split_1/split_dim:output:0*

Tlen0*
T0*M
_output_shapes;
9:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ *
	num_split{
gru_cell/addAddV2gru_cell/split:output:0gru_cell/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ _
gru_cell/SigmoidSigmoidgru_cell/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ }
gru_cell/add_1AddV2gru_cell/split:output:1gru_cell/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ c
gru_cell/Sigmoid_1Sigmoidgru_cell/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ x
gru_cell/mulMulgru_cell/Sigmoid_1:y:0gru_cell/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ t
gru_cell/add_2AddV2gru_cell/split:output:2gru_cell/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ [
gru_cell/TanhTanhgru_cell/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ m
gru_cell/mul_1Mulgru_cell/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:џџџџџџџџџ S
gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?t
gru_cell/subSubgru_cell/sub/x:output:0gru_cell/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ l
gru_cell/mul_2Mulgru_cell/sub:z:0gru_cell/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ q
gru_cell/add_3AddV2gru_cell/mul_1:z:0gru_cell/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :Х
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшвF
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
џџџџџџџџџT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Х
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+gru_cell_readvariableop_gru_8_gru_cell_bias4gru_cell_matmul_readvariableop_gru_8_gru_cell_kernel@gru_cell_matmul_1_readvariableop_gru_8_gru_cell_recurrent_kernel*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :џџџџџџџџџ : : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_50293*
condR
while_cond_50292*8
output_shapes'
%: : : : :џџџџџџџџџ : : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    ж
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:џџџџџџџџџ *
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ 
NoOpNoOp^gru_cell/MatMul/ReadVariableOp!^gru_cell/MatMul_1/ReadVariableOp^gru_cell/ReadVariableOp^while*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:џџџџџџџџџГ: : : 2@
gru_cell/MatMul/ReadVariableOpgru_cell/MatMul/ReadVariableOp2D
 gru_cell/MatMul_1/ReadVariableOp gru_cell/MatMul_1/ReadVariableOp22
gru_cell/ReadVariableOpgru_cell/ReadVariableOp2
whilewhile:?;
9
_user_specified_name!gru_8/gru_cell/recurrent_kernel:51
/
_user_specified_namegru_8/gru_cell/kernel:3/
-
_user_specified_namegru_8/gru_cell/bias:T P
,
_output_shapes
:џџџџџџџџџГ
 
_user_specified_nameinputs


(__inference_gru_cell_layer_call_fn_50567

inputs
states_0%
gru_8_gru_cell_bias:`'
gru_8_gru_cell_kernel:`1
gru_8_gru_cell_recurrent_kernel: `
identity

identity_1ЂStatefulPartitionedCallВ
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0gru_8_gru_cell_biasgru_8_gru_cell_kernelgru_8_gru_cell_recurrent_kernel*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:џџџџџџџџџ :џџџџџџџџџ *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_gru_cell_layer_call_and_return_conditional_losses_49070o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:џџџџџџџџџ:џџџџџџџџџ : : : 22
StatefulPartitionedCallStatefulPartitionedCall:?;
9
_user_specified_name!gru_8/gru_cell/recurrent_kernel:51
/
_user_specified_namegru_8/gru_cell/kernel:3/
-
_user_specified_namegru_8/gru_cell/bias:QM
'
_output_shapes
:џџџџџџџџџ 
"
_user_specified_name
states_0:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ѓ


C__inference_dense_44_layer_call_and_return_conditional_losses_50556

inputs7
%matmul_readvariableop_dense_44_kernel: 2
$biasadd_readvariableop_dense_44_bias:
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOp{
MatMul/ReadVariableOpReadVariableOp%matmul_readvariableop_dense_44_kernel*
_output_shapes

: *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџw
BiasAdd/ReadVariableOpReadVariableOp$biasadd_readvariableop_dense_44_bias*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџV
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџS
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:-)
'
_user_specified_namedense_44/bias:/+
)
_user_specified_namedense_44/kernel:O K
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
й9
К
while_body_50293
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
strided_slice_1_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0?
-gru_cell_readvariableop_gru_8_gru_cell_bias_0:`H
6gru_cell_matmul_readvariableop_gru_8_gru_cell_kernel_0:`T
Bgru_cell_matmul_1_readvariableop_gru_8_gru_cell_recurrent_kernel_0: `
identity

identity_1

identity_2

identity_3

identity_4
strided_slice_1O
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor=
+gru_cell_readvariableop_gru_8_gru_cell_bias:`F
4gru_cell_matmul_readvariableop_gru_8_gru_cell_kernel:`R
@gru_cell_matmul_1_readvariableop_gru_8_gru_cell_recurrent_kernel: `Ђgru_cell/MatMul/ReadVariableOpЂ gru_cell/MatMul_1/ReadVariableOpЂgru_cell/ReadVariableOp
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype0
gru_cell/ReadVariableOpReadVariableOp-gru_cell_readvariableop_gru_8_gru_cell_bias_0*
_output_shapes

:`*
dtype0q
gru_cell/unstackUnpackgru_cell/ReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
num
gru_cell/MatMul/ReadVariableOpReadVariableOp6gru_cell_matmul_readvariableop_gru_8_gru_cell_kernel_0*
_output_shapes

:`*
dtype0
gru_cell/MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0&gru_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ`
gru_cell/BiasAddBiasAddgru_cell/MatMul:product:0gru_cell/unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџ`c
gru_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџО
gru_cell/splitSplit!gru_cell/split/split_dim:output:0gru_cell/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ *
	num_splitЃ
 gru_cell/MatMul_1/ReadVariableOpReadVariableOpBgru_cell_matmul_1_readvariableop_gru_8_gru_cell_recurrent_kernel_0*
_output_shapes

: `*
dtype0
gru_cell/MatMul_1MatMulplaceholder_2(gru_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ`
gru_cell/BiasAdd_1BiasAddgru_cell/MatMul_1:product:0gru_cell/unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџ`c
gru_cell/ConstConst*
_output_shapes
:*
dtype0*!
valueB"        џџџџe
gru_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџъ
gru_cell/split_1SplitVgru_cell/BiasAdd_1:output:0gru_cell/Const:output:0#gru_cell/split_1/split_dim:output:0*

Tlen0*
T0*M
_output_shapes;
9:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ *
	num_split{
gru_cell/addAddV2gru_cell/split:output:0gru_cell/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ _
gru_cell/SigmoidSigmoidgru_cell/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ }
gru_cell/add_1AddV2gru_cell/split:output:1gru_cell/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ c
gru_cell/Sigmoid_1Sigmoidgru_cell/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ x
gru_cell/mulMulgru_cell/Sigmoid_1:y:0gru_cell/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ t
gru_cell/add_2AddV2gru_cell/split:output:2gru_cell/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ [
gru_cell/TanhTanhgru_cell/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ l
gru_cell/mul_1Mulgru_cell/Sigmoid:y:0placeholder_2*
T0*'
_output_shapes
:џџџџџџџџџ S
gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?t
gru_cell/subSubgru_cell/sub/x:output:0gru_cell/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ l
gru_cell/mul_2Mulgru_cell/sub:z:0gru_cell/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ q
gru_cell/add_3AddV2gru_cell/mul_1:z:0gru_cell/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ l
*TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : б
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_13TensorArrayV2Write/TensorListSetItem/index:output:0gru_cell/add_3:z:0*
_output_shapes
: *
element_dtype0:щшвG
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
:џџџџџџџџџ 
NoOpNoOp^gru_cell/MatMul/ReadVariableOp!^gru_cell/MatMul_1/ReadVariableOp^gru_cell/ReadVariableOp*
_output_shapes
 "
@gru_cell_matmul_1_readvariableop_gru_8_gru_cell_recurrent_kernelBgru_cell_matmul_1_readvariableop_gru_8_gru_cell_recurrent_kernel_0"n
4gru_cell_matmul_readvariableop_gru_8_gru_cell_kernel6gru_cell_matmul_readvariableop_gru_8_gru_cell_kernel_0"\
+gru_cell_readvariableop_gru_8_gru_cell_bias-gru_cell_readvariableop_gru_8_gru_cell_bias_0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
identityIdentity:output:0"$
strided_slice_1strided_slice_1_0"
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :џџџџџџџџџ : : : : : 2@
gru_cell/MatMul/ReadVariableOpgru_cell/MatMul/ReadVariableOp2D
 gru_cell/MatMul_1/ReadVariableOp gru_cell/MatMul_1/ReadVariableOp22
gru_cell/ReadVariableOpgru_cell/ReadVariableOp:?	;
9
_user_specified_name!gru_8/gru_cell/recurrent_kernel:51
/
_user_specified_namegru_8/gru_cell/kernel:3/
-
_user_specified_namegru_8/gru_cell/bias:_[
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
:џџџџџџџџџ :
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
і
у
while_cond_50447
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
less_strided_slice_1-
)while_cond_50447___redundant_placeholder0-
)while_cond_50447___redundant_placeholder1-
)while_cond_50447___redundant_placeholder2-
)while_cond_50447___redundant_placeholder3
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
-: : : : :џџџџџџџџџ : :::::
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
:џџџџџџџџџ :
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
і
у
while_cond_49982
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
less_strided_slice_1-
)while_cond_49982___redundant_placeholder0-
)while_cond_49982___redundant_placeholder1-
)while_cond_49982___redundant_placeholder2-
)while_cond_49982___redundant_placeholder3
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
-: : : : :џџџџџџџџџ : :::::
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
:џџџџџџџџџ :
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
"

C__inference_model_52_layer_call_and_return_conditional_losses_49547
input_prior+
gru_8_gru_8_gru_cell_bias:`-
gru_8_gru_8_gru_cell_kernel:`7
%gru_8_gru_8_gru_cell_recurrent_kernel: `*
dense_44_dense_44_kernel: $
dense_44_dense_44_bias:
identityЂ dense_44/StatefulPartitionedCallЂgru_8/StatefulPartitionedCallЫ
masking_prior/PartitionedCallPartitionedCallinput_prior*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџГ
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_masking_prior_layer_call_and_return_conditional_losses_49332
dot_16/PartitionedCallPartitionedCall&masking_prior/PartitionedCall:output:0&masking_prior/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:џџџџџџџџџГГ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dot_16_layer_call_and_return_conditional_losses_49342о
activation_8/PartitionedCallPartitionedCalldot_16/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:џџџџџџџџџГГ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_activation_8_layer_call_and_return_conditional_losses_49348
dot_17/PartitionedCallPartitionedCall%activation_8/PartitionedCall:output:0&masking_prior/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџГ
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dot_17_layer_call_and_return_conditional_losses_49356
multiply_8/PartitionedCallPartitionedCalldot_17/PartitionedCall:output:0&masking_prior/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџГ
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_multiply_8_layer_call_and_return_conditional_losses_49363
concatenate_8/PartitionedCallPartitionedCall&masking_prior/PartitionedCall:output:0#multiply_8/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџГ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_concatenate_8_layer_call_and_return_conditional_losses_49371Ш
gru_8/StatefulPartitionedCallStatefulPartitionedCall&concatenate_8/PartitionedCall:output:0gru_8_gru_8_gru_cell_biasgru_8_gru_8_gru_cell_kernel%gru_8_gru_8_gru_cell_recurrent_kernel*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_gru_8_layer_call_and_return_conditional_losses_49527
 dense_44/StatefulPartitionedCallStatefulPartitionedCall&gru_8/StatefulPartitionedCall:output:0dense_44_dense_44_kerneldense_44_dense_44_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_44_layer_call_and_return_conditional_losses_49542x
IdentityIdentity)dense_44/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџe
NoOpNoOp!^dense_44/StatefulPartitionedCall^gru_8/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":џџџџџџџџџГ
: : : : : 2D
 dense_44/StatefulPartitionedCall dense_44/StatefulPartitionedCall2>
gru_8/StatefulPartitionedCallgru_8/StatefulPartitionedCall:-)
'
_user_specified_namedense_44/bias:/+
)
_user_specified_namedense_44/kernel:?;
9
_user_specified_name!gru_8/gru_cell/recurrent_kernel:51
/
_user_specified_namegru_8/gru_cell/kernel:3/
-
_user_specified_namegru_8/gru_cell/bias:Y U
,
_output_shapes
:џџџџџџџџџГ

%
_user_specified_nameinput_prior
ю6
Ј
@__inference_gru_8_layer_call_and_return_conditional_losses_49143

inputs.
gru_cell_gru_8_gru_cell_bias:`0
gru_cell_gru_8_gru_cell_kernel:`:
(gru_cell_gru_8_gru_cell_recurrent_kernel: `
identityЂ gru_cell/StatefulPartitionedCallЂwhileI
ShapeShapeinputs*
T0*
_output_shapes
::эЯ]
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
valueB:б
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
:џџџџџџџџџ c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџR
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
::эЯ_
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
valueB:л
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
џџџџџџџџџД
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   р
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв_
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
valueB:щ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_maskю
 gru_cell/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0gru_cell_gru_8_gru_cell_biasgru_cell_gru_8_gru_cell_kernel(gru_cell_gru_8_gru_cell_recurrent_kernel*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:џџџџџџџџџ :џџџџџџџџџ *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_gru_cell_layer_call_and_return_conditional_losses_49070n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :Х
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшвF
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
џџџџџџџџџT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0gru_cell_gru_8_gru_cell_biasgru_cell_gru_8_gru_cell_kernel(gru_cell_gru_8_gru_cell_recurrent_kernel*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :џџџџџџџџџ : : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_49081*
condR
while_cond_49080*8
output_shapes'
%: : : : :џџџџџџџџџ : : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    ж
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:џџџџџџџџџ *
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ M
NoOpNoOp!^gru_cell/StatefulPartitionedCall^while*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:џџџџџџџџџџџџџџџџџџ: : : 2D
 gru_cell/StatefulPartitionedCall gru_cell/StatefulPartitionedCall2
whilewhile:?;
9
_user_specified_name!gru_8/gru_cell/recurrent_kernel:51
/
_user_specified_namegru_8/gru_cell/kernel:3/
-
_user_specified_namegru_8/gru_cell/bias:\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
и

C__inference_gru_cell_layer_call_and_return_conditional_losses_49208

inputs

states4
"readvariableop_gru_8_gru_cell_bias:`=
+matmul_readvariableop_gru_8_gru_cell_kernel:`I
7matmul_1_readvariableop_gru_8_gru_cell_recurrent_kernel: `
identity

identity_1ЂMatMul/ReadVariableOpЂMatMul_1/ReadVariableOpЂReadVariableOpq
ReadVariableOpReadVariableOp"readvariableop_gru_8_gru_cell_bias*
_output_shapes

:`*
dtype0_
unstackUnpackReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
num
MatMul/ReadVariableOpReadVariableOp+matmul_readvariableop_gru_8_gru_cell_kernel*
_output_shapes

:`*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ`h
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџ`Z
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџЃ
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ *
	num_split
MatMul_1/ReadVariableOpReadVariableOp7matmul_1_readvariableop_gru_8_gru_cell_recurrent_kernel*
_output_shapes

: `*
dtype0m
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ`l
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџ`Z
ConstConst*
_output_shapes
:*
dtype0*!
valueB"        џџџџ\
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџЦ
split_1SplitVBiasAdd_1:output:0Const:output:0split_1/split_dim:output:0*

Tlen0*
T0*M
_output_shapes;
9:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ *
	num_split`
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ M
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:џџџџџџџџџ b
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ Q
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ ]
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ Y
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ I
TanhTanh	add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ S
mul_1MulSigmoid:y:0states*
T0*'
_output_shapes
:џџџџџџџџџ J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ Q
mul_2Mulsub:z:0Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ V
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ X
IdentityIdentity	add_3:z:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ Z

Identity_1Identity	add_3:z:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ e
NoOpNoOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
_output_shapes
 "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:џџџџџџџџџ:џџџџџџџџџ : : : 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp:?;
9
_user_specified_name!gru_8/gru_cell/recurrent_kernel:51
/
_user_specified_namegru_8/gru_cell/kernel:3/
-
_user_specified_namegru_8/gru_cell/bias:OK
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_namestates:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ж
Љ
(__inference_model_52_layer_call_fn_49739
input_prior%
gru_8_gru_cell_bias:`'
gru_8_gru_cell_kernel:`1
gru_8_gru_cell_recurrent_kernel: `!
dense_44_kernel: 
dense_44_bias:
identityЂStatefulPartitionedCallМ
StatefulPartitionedCallStatefulPartitionedCallinput_priorgru_8_gru_cell_biasgru_8_gru_cell_kernelgru_8_gru_cell_recurrent_kerneldense_44_kerneldense_44_bias*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_model_52_layer_call_and_return_conditional_losses_49719o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":џџџџџџџџџГ
: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:-)
'
_user_specified_namedense_44/bias:/+
)
_user_specified_namedense_44/kernel:?;
9
_user_specified_name!gru_8/gru_cell/recurrent_kernel:51
/
_user_specified_namegru_8/gru_cell/kernel:3/
-
_user_specified_namegru_8/gru_cell/bias:Y U
,
_output_shapes
:џџџџџџџџџГ

%
_user_specified_nameinput_prior
з
r
H__inference_concatenate_8_layer_call_and_return_conditional_losses_49371

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
:џџџџџџџџџГ\
IdentityIdentityconcat:output:0*
T0*,
_output_shapes
:џџџџџџџџџГ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:џџџџџџџџџГ
:џџџџџџџџџГ
:TP
,
_output_shapes
:џџџџџџџџџГ

 
_user_specified_nameinputs:T P
,
_output_shapes
:џџџџџџџџџГ

 
_user_specified_nameinputs
Б!

while_body_49081
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
strided_slice_1_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_00
gru_cell_gru_8_gru_cell_bias_0:`2
 gru_cell_gru_8_gru_cell_kernel_0:`<
*gru_cell_gru_8_gru_cell_recurrent_kernel_0: `
identity

identity_1

identity_2

identity_3

identity_4
strided_slice_1O
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor.
gru_cell_gru_8_gru_cell_bias:`0
gru_cell_gru_8_gru_cell_kernel:`:
(gru_cell_gru_8_gru_cell_recurrent_kernel: `Ђ gru_cell/StatefulPartitionedCall
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype0
 gru_cell/StatefulPartitionedCallStatefulPartitionedCall*TensorArrayV2Read/TensorListGetItem:item:0placeholder_2gru_cell_gru_8_gru_cell_bias_0 gru_cell_gru_8_gru_cell_kernel_0*gru_cell_gru_8_gru_cell_recurrent_kernel_0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:џџџџџџџџџ :џџџџџџџџџ *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_gru_cell_layer_call_and_return_conditional_losses_49070l
*TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : ш
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_13TensorArrayV2Write/TensorListSetItem/index:output:0)gru_cell/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:щшвG
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
:џџџџџџџџџ E
NoOpNoOp!^gru_cell/StatefulPartitionedCall*
_output_shapes
 ">
gru_cell_gru_8_gru_cell_biasgru_cell_gru_8_gru_cell_bias_0"B
gru_cell_gru_8_gru_cell_kernel gru_cell_gru_8_gru_cell_kernel_0"V
(gru_cell_gru_8_gru_cell_recurrent_kernel*gru_cell_gru_8_gru_cell_recurrent_kernel_0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
identityIdentity:output:0"$
strided_slice_1strided_slice_1_0"
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :џџџџџџџџџ : : : : : 2D
 gru_cell/StatefulPartitionedCall gru_cell/StatefulPartitionedCall:?	;
9
_user_specified_name!gru_8/gru_cell/recurrent_kernel:51
/
_user_specified_namegru_8/gru_cell/kernel:3/
-
_user_specified_namegru_8/gru_cell/bias:_[
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
:џџџџџџџџџ :
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

Є
#__inference_signature_wrapper_49807
input_prior%
gru_8_gru_cell_bias:`'
gru_8_gru_cell_kernel:`1
gru_8_gru_cell_recurrent_kernel: `!
dense_44_kernel: 
dense_44_bias:
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_priorgru_8_gru_cell_biasgru_8_gru_cell_kernelgru_8_gru_cell_recurrent_kerneldense_44_kerneldense_44_bias*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8 *)
f$R"
 __inference__wrapped_model_49005o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":џџџџџџџџџГ
: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:-)
'
_user_specified_namedense_44/bias:/+
)
_user_specified_namedense_44/kernel:?;
9
_user_specified_name!gru_8/gru_cell/recurrent_kernel:51
/
_user_specified_namegru_8/gru_cell/kernel:3/
-
_user_specified_namegru_8/gru_cell/bias:Y U
,
_output_shapes
:џџџџџџџџџГ

%
_user_specified_nameinput_prior
"

C__inference_model_52_layer_call_and_return_conditional_losses_49719
input_prior+
gru_8_gru_8_gru_cell_bias:`-
gru_8_gru_8_gru_cell_kernel:`7
%gru_8_gru_8_gru_cell_recurrent_kernel: `*
dense_44_dense_44_kernel: $
dense_44_dense_44_bias:
identityЂ dense_44/StatefulPartitionedCallЂgru_8/StatefulPartitionedCallЫ
masking_prior/PartitionedCallPartitionedCallinput_prior*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџГ
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_masking_prior_layer_call_and_return_conditional_losses_49332
dot_16/PartitionedCallPartitionedCall&masking_prior/PartitionedCall:output:0&masking_prior/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:џџџџџџџџџГГ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dot_16_layer_call_and_return_conditional_losses_49342о
activation_8/PartitionedCallPartitionedCalldot_16/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:џџџџџџџџџГГ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_activation_8_layer_call_and_return_conditional_losses_49348
dot_17/PartitionedCallPartitionedCall%activation_8/PartitionedCall:output:0&masking_prior/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџГ
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dot_17_layer_call_and_return_conditional_losses_49356
multiply_8/PartitionedCallPartitionedCalldot_17/PartitionedCall:output:0&masking_prior/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџГ
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_multiply_8_layer_call_and_return_conditional_losses_49363
concatenate_8/PartitionedCallPartitionedCall&masking_prior/PartitionedCall:output:0#multiply_8/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџГ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_concatenate_8_layer_call_and_return_conditional_losses_49371Ш
gru_8/StatefulPartitionedCallStatefulPartitionedCall&concatenate_8/PartitionedCall:output:0gru_8_gru_8_gru_cell_biasgru_8_gru_8_gru_cell_kernel%gru_8_gru_8_gru_cell_recurrent_kernel*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_gru_8_layer_call_and_return_conditional_losses_49710
 dense_44/StatefulPartitionedCallStatefulPartitionedCall&gru_8/StatefulPartitionedCall:output:0dense_44_dense_44_kerneldense_44_dense_44_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_44_layer_call_and_return_conditional_losses_49542x
IdentityIdentity)dense_44/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџe
NoOpNoOp!^dense_44/StatefulPartitionedCall^gru_8/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":џџџџџџџџџГ
: : : : : 2D
 dense_44/StatefulPartitionedCall dense_44/StatefulPartitionedCall2>
gru_8/StatefulPartitionedCallgru_8/StatefulPartitionedCall:-)
'
_user_specified_namedense_44/bias:/+
)
_user_specified_namedense_44/kernel:?;
9
_user_specified_name!gru_8/gru_cell/recurrent_kernel:51
/
_user_specified_namegru_8/gru_cell/kernel:3/
-
_user_specified_namegru_8/gru_cell/bias:Y U
,
_output_shapes
:џџџџџџџџџГ

%
_user_specified_nameinput_prior
Л
H
,__inference_activation_8_layer_call_fn_49843

inputs
identityИ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:џџџџџџџџџГГ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_activation_8_layer_call_and_return_conditional_losses_49348f
IdentityIdentityPartitionedCall:output:0*
T0*-
_output_shapes
:џџџџџџџџџГГ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:џџџџџџџџџГГ:U Q
-
_output_shapes
:џџџџџџџџџГГ
 
_user_specified_nameinputs
Т
V
*__inference_multiply_8_layer_call_fn_49867
inputs_0
inputs_1
identityТ
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџГ
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_multiply_8_layer_call_and_return_conditional_losses_49363e
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:џџџџџџџџџГ
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:џџџџџџџџџГ
:џџџџџџџџџГ
:VR
,
_output_shapes
:џџџџџџџџџГ

"
_user_specified_name
inputs_1:V R
,
_output_shapes
:џџџџџџџџџГ

"
_user_specified_name
inputs_0
ВN
 
@__inference_gru_8_layer_call_and_return_conditional_losses_49527

inputs=
+gru_cell_readvariableop_gru_8_gru_cell_bias:`F
4gru_cell_matmul_readvariableop_gru_8_gru_cell_kernel:`R
@gru_cell_matmul_1_readvariableop_gru_8_gru_cell_recurrent_kernel: `
identityЂgru_cell/MatMul/ReadVariableOpЂ gru_cell/MatMul_1/ReadVariableOpЂgru_cell/ReadVariableOpЂwhileI
ShapeShapeinputs*
T0*
_output_shapes
::эЯ]
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
valueB:б
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
:џџџџџџџџџ c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          n
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:ГџџџџџџџџџR
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
::эЯ_
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
valueB:л
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
џџџџџџџџџД
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   р
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв_
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
valueB:щ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask
gru_cell/ReadVariableOpReadVariableOp+gru_cell_readvariableop_gru_8_gru_cell_bias*
_output_shapes

:`*
dtype0q
gru_cell/unstackUnpackgru_cell/ReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
num
gru_cell/MatMul/ReadVariableOpReadVariableOp4gru_cell_matmul_readvariableop_gru_8_gru_cell_kernel*
_output_shapes

:`*
dtype0
gru_cell/MatMulMatMulstrided_slice_2:output:0&gru_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ`
gru_cell/BiasAddBiasAddgru_cell/MatMul:product:0gru_cell/unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџ`c
gru_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџО
gru_cell/splitSplit!gru_cell/split/split_dim:output:0gru_cell/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ *
	num_splitЁ
 gru_cell/MatMul_1/ReadVariableOpReadVariableOp@gru_cell_matmul_1_readvariableop_gru_8_gru_cell_recurrent_kernel*
_output_shapes

: `*
dtype0
gru_cell/MatMul_1MatMulzeros:output:0(gru_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ`
gru_cell/BiasAdd_1BiasAddgru_cell/MatMul_1:product:0gru_cell/unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџ`c
gru_cell/ConstConst*
_output_shapes
:*
dtype0*!
valueB"        џџџџe
gru_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџъ
gru_cell/split_1SplitVgru_cell/BiasAdd_1:output:0gru_cell/Const:output:0#gru_cell/split_1/split_dim:output:0*

Tlen0*
T0*M
_output_shapes;
9:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ *
	num_split{
gru_cell/addAddV2gru_cell/split:output:0gru_cell/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ _
gru_cell/SigmoidSigmoidgru_cell/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ }
gru_cell/add_1AddV2gru_cell/split:output:1gru_cell/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ c
gru_cell/Sigmoid_1Sigmoidgru_cell/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ x
gru_cell/mulMulgru_cell/Sigmoid_1:y:0gru_cell/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ t
gru_cell/add_2AddV2gru_cell/split:output:2gru_cell/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ [
gru_cell/TanhTanhgru_cell/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ m
gru_cell/mul_1Mulgru_cell/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:џџџџџџџџџ S
gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?t
gru_cell/subSubgru_cell/sub/x:output:0gru_cell/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ l
gru_cell/mul_2Mulgru_cell/sub:z:0gru_cell/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ q
gru_cell/add_3AddV2gru_cell/mul_1:z:0gru_cell/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :Х
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшвF
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
џџџџџџџџџT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Х
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+gru_cell_readvariableop_gru_8_gru_cell_bias4gru_cell_matmul_readvariableop_gru_8_gru_cell_kernel@gru_cell_matmul_1_readvariableop_gru_8_gru_cell_recurrent_kernel*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :џџџџџџџџџ : : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_49437*
condR
while_cond_49436*8
output_shapes'
%: : : : :џџџџџџџџџ : : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    ж
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:џџџџџџџџџ *
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ 
NoOpNoOp^gru_cell/MatMul/ReadVariableOp!^gru_cell/MatMul_1/ReadVariableOp^gru_cell/ReadVariableOp^while*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:џџџџџџџџџГ: : : 2@
gru_cell/MatMul/ReadVariableOpgru_cell/MatMul/ReadVariableOp2D
 gru_cell/MatMul_1/ReadVariableOp gru_cell/MatMul_1/ReadVariableOp22
gru_cell/ReadVariableOpgru_cell/ReadVariableOp2
whilewhile:?;
9
_user_specified_name!gru_8/gru_cell/recurrent_kernel:51
/
_user_specified_namegru_8/gru_cell/kernel:3/
-
_user_specified_namegru_8/gru_cell/bias:T P
,
_output_shapes
:џџџџџџџџџГ
 
_user_specified_nameinputs
Ъ
d
H__inference_masking_prior_layer_call_and_return_conditional_losses_49332

inputs
identityO

NotEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  Пh
NotEqualNotEqualinputsNotEqual/y:output:0*
T0*,
_output_shapes
:џџџџџџџџџГ
`
Any/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџw
AnyAnyNotEqual:z:0Any/reduction_indices:output:0*,
_output_shapes
:џџџџџџџџџГ*
	keep_dims(`
CastCastAny:output:0*

DstT0*

SrcT0
*,
_output_shapes
:џџџџџџџџџГS
mulMulinputsCast:y:0*
T0*,
_output_shapes
:џџџџџџџџџГ
s
SqueezeSqueezeAny:output:0*
T0
*(
_output_shapes
:џџџџџџџџџГ*
squeeze_dims

џџџџџџџџџT
IdentityIdentitymul:z:0*
T0*,
_output_shapes
:џџџџџџџџџГ
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџГ
:T P
,
_output_shapes
:џџџџџџџџџГ

 
_user_specified_nameinputs
й9
К
while_body_49983
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
strided_slice_1_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0?
-gru_cell_readvariableop_gru_8_gru_cell_bias_0:`H
6gru_cell_matmul_readvariableop_gru_8_gru_cell_kernel_0:`T
Bgru_cell_matmul_1_readvariableop_gru_8_gru_cell_recurrent_kernel_0: `
identity

identity_1

identity_2

identity_3

identity_4
strided_slice_1O
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor=
+gru_cell_readvariableop_gru_8_gru_cell_bias:`F
4gru_cell_matmul_readvariableop_gru_8_gru_cell_kernel:`R
@gru_cell_matmul_1_readvariableop_gru_8_gru_cell_recurrent_kernel: `Ђgru_cell/MatMul/ReadVariableOpЂ gru_cell/MatMul_1/ReadVariableOpЂgru_cell/ReadVariableOp
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype0
gru_cell/ReadVariableOpReadVariableOp-gru_cell_readvariableop_gru_8_gru_cell_bias_0*
_output_shapes

:`*
dtype0q
gru_cell/unstackUnpackgru_cell/ReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
num
gru_cell/MatMul/ReadVariableOpReadVariableOp6gru_cell_matmul_readvariableop_gru_8_gru_cell_kernel_0*
_output_shapes

:`*
dtype0
gru_cell/MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0&gru_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ`
gru_cell/BiasAddBiasAddgru_cell/MatMul:product:0gru_cell/unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџ`c
gru_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџО
gru_cell/splitSplit!gru_cell/split/split_dim:output:0gru_cell/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ *
	num_splitЃ
 gru_cell/MatMul_1/ReadVariableOpReadVariableOpBgru_cell_matmul_1_readvariableop_gru_8_gru_cell_recurrent_kernel_0*
_output_shapes

: `*
dtype0
gru_cell/MatMul_1MatMulplaceholder_2(gru_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ`
gru_cell/BiasAdd_1BiasAddgru_cell/MatMul_1:product:0gru_cell/unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџ`c
gru_cell/ConstConst*
_output_shapes
:*
dtype0*!
valueB"        џџџџe
gru_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџъ
gru_cell/split_1SplitVgru_cell/BiasAdd_1:output:0gru_cell/Const:output:0#gru_cell/split_1/split_dim:output:0*

Tlen0*
T0*M
_output_shapes;
9:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ *
	num_split{
gru_cell/addAddV2gru_cell/split:output:0gru_cell/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ _
gru_cell/SigmoidSigmoidgru_cell/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ }
gru_cell/add_1AddV2gru_cell/split:output:1gru_cell/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ c
gru_cell/Sigmoid_1Sigmoidgru_cell/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ x
gru_cell/mulMulgru_cell/Sigmoid_1:y:0gru_cell/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ t
gru_cell/add_2AddV2gru_cell/split:output:2gru_cell/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ [
gru_cell/TanhTanhgru_cell/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ l
gru_cell/mul_1Mulgru_cell/Sigmoid:y:0placeholder_2*
T0*'
_output_shapes
:џџџџџџџџџ S
gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?t
gru_cell/subSubgru_cell/sub/x:output:0gru_cell/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ l
gru_cell/mul_2Mulgru_cell/sub:z:0gru_cell/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ q
gru_cell/add_3AddV2gru_cell/mul_1:z:0gru_cell/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ l
*TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : б
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_13TensorArrayV2Write/TensorListSetItem/index:output:0gru_cell/add_3:z:0*
_output_shapes
: *
element_dtype0:щшвG
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
:џџџџџџџџџ 
NoOpNoOp^gru_cell/MatMul/ReadVariableOp!^gru_cell/MatMul_1/ReadVariableOp^gru_cell/ReadVariableOp*
_output_shapes
 "
@gru_cell_matmul_1_readvariableop_gru_8_gru_cell_recurrent_kernelBgru_cell_matmul_1_readvariableop_gru_8_gru_cell_recurrent_kernel_0"n
4gru_cell_matmul_readvariableop_gru_8_gru_cell_kernel6gru_cell_matmul_readvariableop_gru_8_gru_cell_kernel_0"\
+gru_cell_readvariableop_gru_8_gru_cell_bias-gru_cell_readvariableop_gru_8_gru_cell_bias_0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
identityIdentity:output:0"$
strided_slice_1strided_slice_1_0"
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :џџџџџџџџџ : : : : : 2@
gru_cell/MatMul/ReadVariableOpgru_cell/MatMul/ReadVariableOp2D
 gru_cell/MatMul_1/ReadVariableOp gru_cell/MatMul_1/ReadVariableOp22
gru_cell/ReadVariableOpgru_cell/ReadVariableOp:?	;
9
_user_specified_name!gru_8/gru_cell/recurrent_kernel:51
/
_user_specified_namegru_8/gru_cell/kernel:3/
-
_user_specified_namegru_8/gru_cell/bias:_[
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
:џџџџџџџџџ :
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
р

C__inference_gru_cell_layer_call_and_return_conditional_losses_50656

inputs
states_04
"readvariableop_gru_8_gru_cell_bias:`=
+matmul_readvariableop_gru_8_gru_cell_kernel:`I
7matmul_1_readvariableop_gru_8_gru_cell_recurrent_kernel: `
identity

identity_1ЂMatMul/ReadVariableOpЂMatMul_1/ReadVariableOpЂReadVariableOpq
ReadVariableOpReadVariableOp"readvariableop_gru_8_gru_cell_bias*
_output_shapes

:`*
dtype0_
unstackUnpackReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
num
MatMul/ReadVariableOpReadVariableOp+matmul_readvariableop_gru_8_gru_cell_kernel*
_output_shapes

:`*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ`h
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџ`Z
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџЃ
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ *
	num_split
MatMul_1/ReadVariableOpReadVariableOp7matmul_1_readvariableop_gru_8_gru_cell_recurrent_kernel*
_output_shapes

: `*
dtype0o
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ`l
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџ`Z
ConstConst*
_output_shapes
:*
dtype0*!
valueB"        џџџџ\
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџЦ
split_1SplitVBiasAdd_1:output:0Const:output:0split_1/split_dim:output:0*

Tlen0*
T0*M
_output_shapes;
9:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ *
	num_split`
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ M
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:џџџџџџџџџ b
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ Q
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ ]
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ Y
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ I
TanhTanh	add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ U
mul_1MulSigmoid:y:0states_0*
T0*'
_output_shapes
:џџџџџџџџџ J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ Q
mul_2Mulsub:z:0Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ V
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ X
IdentityIdentity	add_3:z:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ Z

Identity_1Identity	add_3:z:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ e
NoOpNoOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
_output_shapes
 "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:џџџџџџџџџ:џџџџџџџџџ : : : 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp:?;
9
_user_specified_name!gru_8/gru_cell/recurrent_kernel:51
/
_user_specified_namegru_8/gru_cell/kernel:3/
-
_user_specified_namegru_8/gru_cell/bias:QM
'
_output_shapes
:џџџџџџџџџ 
"
_user_specified_name
states_0:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
М
R
&__inference_dot_16_layer_call_fn_49829
inputs_0
inputs_1
identityП
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:џџџџџџџџџГГ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dot_16_layer_call_and_return_conditional_losses_49342f
IdentityIdentityPartitionedCall:output:0*
T0*-
_output_shapes
:џџџџџџџџџГГ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:џџџџџџџџџГ
:џџџџџџџџџГ
:VR
,
_output_shapes
:џџџџџџџџџГ

"
_user_specified_name
inputs_1:V R
,
_output_shapes
:џџџџџџџџџГ

"
_user_specified_name
inputs_0
й	
с
%__inference_gru_8_layer_call_fn_49910

inputs%
gru_8_gru_cell_bias:`'
gru_8_gru_cell_kernel:`1
gru_8_gru_cell_recurrent_kernel: `
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsgru_8_gru_cell_biasgru_8_gru_cell_kernelgru_8_gru_cell_recurrent_kernel*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_gru_8_layer_call_and_return_conditional_losses_49527o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:џџџџџџџџџГ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:?;
9
_user_specified_name!gru_8/gru_cell/recurrent_kernel:51
/
_user_specified_namegru_8/gru_cell/kernel:3/
-
_user_specified_namegru_8/gru_cell/bias:T P
,
_output_shapes
:џџџџџџџџџГ
 
_user_specified_nameinputs
я	
у
%__inference_gru_8_layer_call_fn_49894
inputs_0%
gru_8_gru_cell_bias:`'
gru_8_gru_cell_kernel:`1
gru_8_gru_cell_recurrent_kernel: `
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputs_0gru_8_gru_cell_biasgru_8_gru_cell_kernelgru_8_gru_cell_recurrent_kernel*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_gru_8_layer_call_and_return_conditional_losses_49143o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:џџџџџџџџџџџџџџџџџџ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:?;
9
_user_specified_name!gru_8/gru_cell/recurrent_kernel:51
/
_user_specified_namegru_8/gru_cell/kernel:3/
-
_user_specified_namegru_8/gru_cell/bias:^ Z
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs_0
й9
К
while_body_49437
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
strided_slice_1_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0?
-gru_cell_readvariableop_gru_8_gru_cell_bias_0:`H
6gru_cell_matmul_readvariableop_gru_8_gru_cell_kernel_0:`T
Bgru_cell_matmul_1_readvariableop_gru_8_gru_cell_recurrent_kernel_0: `
identity

identity_1

identity_2

identity_3

identity_4
strided_slice_1O
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor=
+gru_cell_readvariableop_gru_8_gru_cell_bias:`F
4gru_cell_matmul_readvariableop_gru_8_gru_cell_kernel:`R
@gru_cell_matmul_1_readvariableop_gru_8_gru_cell_recurrent_kernel: `Ђgru_cell/MatMul/ReadVariableOpЂ gru_cell/MatMul_1/ReadVariableOpЂgru_cell/ReadVariableOp
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype0
gru_cell/ReadVariableOpReadVariableOp-gru_cell_readvariableop_gru_8_gru_cell_bias_0*
_output_shapes

:`*
dtype0q
gru_cell/unstackUnpackgru_cell/ReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
num
gru_cell/MatMul/ReadVariableOpReadVariableOp6gru_cell_matmul_readvariableop_gru_8_gru_cell_kernel_0*
_output_shapes

:`*
dtype0
gru_cell/MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0&gru_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ`
gru_cell/BiasAddBiasAddgru_cell/MatMul:product:0gru_cell/unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџ`c
gru_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџО
gru_cell/splitSplit!gru_cell/split/split_dim:output:0gru_cell/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ *
	num_splitЃ
 gru_cell/MatMul_1/ReadVariableOpReadVariableOpBgru_cell_matmul_1_readvariableop_gru_8_gru_cell_recurrent_kernel_0*
_output_shapes

: `*
dtype0
gru_cell/MatMul_1MatMulplaceholder_2(gru_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ`
gru_cell/BiasAdd_1BiasAddgru_cell/MatMul_1:product:0gru_cell/unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџ`c
gru_cell/ConstConst*
_output_shapes
:*
dtype0*!
valueB"        џџџџe
gru_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџъ
gru_cell/split_1SplitVgru_cell/BiasAdd_1:output:0gru_cell/Const:output:0#gru_cell/split_1/split_dim:output:0*

Tlen0*
T0*M
_output_shapes;
9:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ *
	num_split{
gru_cell/addAddV2gru_cell/split:output:0gru_cell/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ _
gru_cell/SigmoidSigmoidgru_cell/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ }
gru_cell/add_1AddV2gru_cell/split:output:1gru_cell/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ c
gru_cell/Sigmoid_1Sigmoidgru_cell/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ x
gru_cell/mulMulgru_cell/Sigmoid_1:y:0gru_cell/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ t
gru_cell/add_2AddV2gru_cell/split:output:2gru_cell/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ [
gru_cell/TanhTanhgru_cell/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ l
gru_cell/mul_1Mulgru_cell/Sigmoid:y:0placeholder_2*
T0*'
_output_shapes
:џџџџџџџџџ S
gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?t
gru_cell/subSubgru_cell/sub/x:output:0gru_cell/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ l
gru_cell/mul_2Mulgru_cell/sub:z:0gru_cell/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ q
gru_cell/add_3AddV2gru_cell/mul_1:z:0gru_cell/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ l
*TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : б
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_13TensorArrayV2Write/TensorListSetItem/index:output:0gru_cell/add_3:z:0*
_output_shapes
: *
element_dtype0:щшвG
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
:џџџџџџџџџ 
NoOpNoOp^gru_cell/MatMul/ReadVariableOp!^gru_cell/MatMul_1/ReadVariableOp^gru_cell/ReadVariableOp*
_output_shapes
 "
@gru_cell_matmul_1_readvariableop_gru_8_gru_cell_recurrent_kernelBgru_cell_matmul_1_readvariableop_gru_8_gru_cell_recurrent_kernel_0"n
4gru_cell_matmul_readvariableop_gru_8_gru_cell_kernel6gru_cell_matmul_readvariableop_gru_8_gru_cell_kernel_0"\
+gru_cell_readvariableop_gru_8_gru_cell_bias-gru_cell_readvariableop_gru_8_gru_cell_bias_0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
identityIdentity:output:0"$
strided_slice_1strided_slice_1_0"
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :џџџџџџџџџ : : : : : 2@
gru_cell/MatMul/ReadVariableOpgru_cell/MatMul/ReadVariableOp2D
 gru_cell/MatMul_1/ReadVariableOp gru_cell/MatMul_1/ReadVariableOp22
gru_cell/ReadVariableOpgru_cell/ReadVariableOp:?	;
9
_user_specified_name!gru_8/gru_cell/recurrent_kernel:51
/
_user_specified_namegru_8/gru_cell/kernel:3/
-
_user_specified_namegru_8/gru_cell/bias:_[
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
:џџџџџџџџџ :
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
о
q
E__inference_multiply_8_layer_call_and_return_conditional_losses_49873
inputs_0
inputs_1
identityU
mulMulinputs_0inputs_1*
T0*,
_output_shapes
:џџџџџџџџџГ
T
IdentityIdentitymul:z:0*
T0*,
_output_shapes
:џџџџџџџџџГ
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:џџџџџџџџџГ
:џџџџџџџџџГ
:VR
,
_output_shapes
:џџџџџџџџџГ

"
_user_specified_name
inputs_1:V R
,
_output_shapes
:џџџџџџџџџГ

"
_user_specified_name
inputs_0
і
у
while_cond_50137
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
less_strided_slice_1-
)while_cond_50137___redundant_placeholder0-
)while_cond_50137___redundant_placeholder1-
)while_cond_50137___redundant_placeholder2-
)while_cond_50137___redundant_placeholder3
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
-: : : : :џџџџџџџџџ : :::::
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
:џџџџџџџџџ :
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
уг
Д
__inference__traced_save_50834
file_prefix8
&read_disablecopyonread_dense_44_kernel: 4
&read_1_disablecopyonread_dense_44_bias:@
.read_2_disablecopyonread_gru_8_gru_cell_kernel:`J
8read_3_disablecopyonread_gru_8_gru_cell_recurrent_kernel: `>
,read_4_disablecopyonread_gru_8_gru_cell_bias:`8
.read_5_disablecopyonread_training_16_adam_iter:	 :
0read_6_disablecopyonread_training_16_adam_beta_1: :
0read_7_disablecopyonread_training_16_adam_beta_2: 9
/read_8_disablecopyonread_training_16_adam_decay: A
7read_9_disablecopyonread_training_16_adam_learning_rate: +
!read_10_disablecopyonread_total_8: +
!read_11_disablecopyonread_count_8: 9
*read_12_disablecopyonread_true_positives_8:	Ш9
*read_13_disablecopyonread_true_negatives_8:	Ш:
+read_14_disablecopyonread_false_positives_8:	Ш:
+read_15_disablecopyonread_false_negatives_8:	ШN
<read_16_disablecopyonread_training_16_adam_dense_44_kernel_m: H
:read_17_disablecopyonread_training_16_adam_dense_44_bias_m:T
Bread_18_disablecopyonread_training_16_adam_gru_8_gru_cell_kernel_m:`^
Lread_19_disablecopyonread_training_16_adam_gru_8_gru_cell_recurrent_kernel_m: `R
@read_20_disablecopyonread_training_16_adam_gru_8_gru_cell_bias_m:`N
<read_21_disablecopyonread_training_16_adam_dense_44_kernel_v: H
:read_22_disablecopyonread_training_16_adam_dense_44_bias_v:T
Bread_23_disablecopyonread_training_16_adam_gru_8_gru_cell_kernel_v:`^
Lread_24_disablecopyonread_training_16_adam_gru_8_gru_cell_recurrent_kernel_v: `R
@read_25_disablecopyonread_training_16_adam_gru_8_gru_cell_bias_v:`
savev2_const
identity_53ЂMergeV2CheckpointsЂRead/DisableCopyOnReadЂRead/ReadVariableOpЂRead_1/DisableCopyOnReadЂRead_1/ReadVariableOpЂRead_10/DisableCopyOnReadЂRead_10/ReadVariableOpЂRead_11/DisableCopyOnReadЂRead_11/ReadVariableOpЂRead_12/DisableCopyOnReadЂRead_12/ReadVariableOpЂRead_13/DisableCopyOnReadЂRead_13/ReadVariableOpЂRead_14/DisableCopyOnReadЂRead_14/ReadVariableOpЂRead_15/DisableCopyOnReadЂRead_15/ReadVariableOpЂRead_16/DisableCopyOnReadЂRead_16/ReadVariableOpЂRead_17/DisableCopyOnReadЂRead_17/ReadVariableOpЂRead_18/DisableCopyOnReadЂRead_18/ReadVariableOpЂRead_19/DisableCopyOnReadЂRead_19/ReadVariableOpЂRead_2/DisableCopyOnReadЂRead_2/ReadVariableOpЂRead_20/DisableCopyOnReadЂRead_20/ReadVariableOpЂRead_21/DisableCopyOnReadЂRead_21/ReadVariableOpЂRead_22/DisableCopyOnReadЂRead_22/ReadVariableOpЂRead_23/DisableCopyOnReadЂRead_23/ReadVariableOpЂRead_24/DisableCopyOnReadЂRead_24/ReadVariableOpЂRead_25/DisableCopyOnReadЂRead_25/ReadVariableOpЂRead_3/DisableCopyOnReadЂRead_3/ReadVariableOpЂRead_4/DisableCopyOnReadЂRead_4/ReadVariableOpЂRead_5/DisableCopyOnReadЂRead_5/ReadVariableOpЂRead_6/DisableCopyOnReadЂRead_6/ReadVariableOpЂRead_7/DisableCopyOnReadЂRead_7/ReadVariableOpЂRead_8/DisableCopyOnReadЂRead_8/ReadVariableOpЂRead_9/DisableCopyOnReadЂRead_9/ReadVariableOpw
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
_temp/part
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
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: x
Read/DisableCopyOnReadDisableCopyOnRead&read_disablecopyonread_dense_44_kernel"/device:CPU:0*
_output_shapes
 Ђ
Read/ReadVariableOpReadVariableOp&read_disablecopyonread_dense_44_kernel^Read/DisableCopyOnRead"/device:CPU:0*
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

: z
Read_1/DisableCopyOnReadDisableCopyOnRead&read_1_disablecopyonread_dense_44_bias"/device:CPU:0*
_output_shapes
 Ђ
Read_1/ReadVariableOpReadVariableOp&read_1_disablecopyonread_dense_44_bias^Read_1/DisableCopyOnRead"/device:CPU:0*
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
:
Read_2/DisableCopyOnReadDisableCopyOnRead.read_2_disablecopyonread_gru_8_gru_cell_kernel"/device:CPU:0*
_output_shapes
 Ў
Read_2/ReadVariableOpReadVariableOp.read_2_disablecopyonread_gru_8_gru_cell_kernel^Read_2/DisableCopyOnRead"/device:CPU:0*
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

:`
Read_3/DisableCopyOnReadDisableCopyOnRead8read_3_disablecopyonread_gru_8_gru_cell_recurrent_kernel"/device:CPU:0*
_output_shapes
 И
Read_3/ReadVariableOpReadVariableOp8read_3_disablecopyonread_gru_8_gru_cell_recurrent_kernel^Read_3/DisableCopyOnRead"/device:CPU:0*
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

: `
Read_4/DisableCopyOnReadDisableCopyOnRead,read_4_disablecopyonread_gru_8_gru_cell_bias"/device:CPU:0*
_output_shapes
 Ќ
Read_4/ReadVariableOpReadVariableOp,read_4_disablecopyonread_gru_8_gru_cell_bias^Read_4/DisableCopyOnRead"/device:CPU:0*
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

:`
Read_5/DisableCopyOnReadDisableCopyOnRead.read_5_disablecopyonread_training_16_adam_iter"/device:CPU:0*
_output_shapes
 І
Read_5/ReadVariableOpReadVariableOp.read_5_disablecopyonread_training_16_adam_iter^Read_5/DisableCopyOnRead"/device:CPU:0*
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
: 
Read_6/DisableCopyOnReadDisableCopyOnRead0read_6_disablecopyonread_training_16_adam_beta_1"/device:CPU:0*
_output_shapes
 Ј
Read_6/ReadVariableOpReadVariableOp0read_6_disablecopyonread_training_16_adam_beta_1^Read_6/DisableCopyOnRead"/device:CPU:0*
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
: 
Read_7/DisableCopyOnReadDisableCopyOnRead0read_7_disablecopyonread_training_16_adam_beta_2"/device:CPU:0*
_output_shapes
 Ј
Read_7/ReadVariableOpReadVariableOp0read_7_disablecopyonread_training_16_adam_beta_2^Read_7/DisableCopyOnRead"/device:CPU:0*
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
: 
Read_8/DisableCopyOnReadDisableCopyOnRead/read_8_disablecopyonread_training_16_adam_decay"/device:CPU:0*
_output_shapes
 Ї
Read_8/ReadVariableOpReadVariableOp/read_8_disablecopyonread_training_16_adam_decay^Read_8/DisableCopyOnRead"/device:CPU:0*
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
: 
Read_9/DisableCopyOnReadDisableCopyOnRead7read_9_disablecopyonread_training_16_adam_learning_rate"/device:CPU:0*
_output_shapes
 Џ
Read_9/ReadVariableOpReadVariableOp7read_9_disablecopyonread_training_16_adam_learning_rate^Read_9/DisableCopyOnRead"/device:CPU:0*
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
: v
Read_10/DisableCopyOnReadDisableCopyOnRead!read_10_disablecopyonread_total_8"/device:CPU:0*
_output_shapes
 
Read_10/ReadVariableOpReadVariableOp!read_10_disablecopyonread_total_8^Read_10/DisableCopyOnRead"/device:CPU:0*
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
: v
Read_11/DisableCopyOnReadDisableCopyOnRead!read_11_disablecopyonread_count_8"/device:CPU:0*
_output_shapes
 
Read_11/ReadVariableOpReadVariableOp!read_11_disablecopyonread_count_8^Read_11/DisableCopyOnRead"/device:CPU:0*
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
: 
Read_12/DisableCopyOnReadDisableCopyOnRead*read_12_disablecopyonread_true_positives_8"/device:CPU:0*
_output_shapes
 Љ
Read_12/ReadVariableOpReadVariableOp*read_12_disablecopyonread_true_positives_8^Read_12/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:Ш*
dtype0l
Identity_24IdentityRead_12/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:Шb
Identity_25IdentityIdentity_24:output:0"/device:CPU:0*
T0*
_output_shapes	
:Ш
Read_13/DisableCopyOnReadDisableCopyOnRead*read_13_disablecopyonread_true_negatives_8"/device:CPU:0*
_output_shapes
 Љ
Read_13/ReadVariableOpReadVariableOp*read_13_disablecopyonread_true_negatives_8^Read_13/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:Ш*
dtype0l
Identity_26IdentityRead_13/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:Шb
Identity_27IdentityIdentity_26:output:0"/device:CPU:0*
T0*
_output_shapes	
:Ш
Read_14/DisableCopyOnReadDisableCopyOnRead+read_14_disablecopyonread_false_positives_8"/device:CPU:0*
_output_shapes
 Њ
Read_14/ReadVariableOpReadVariableOp+read_14_disablecopyonread_false_positives_8^Read_14/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:Ш*
dtype0l
Identity_28IdentityRead_14/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:Шb
Identity_29IdentityIdentity_28:output:0"/device:CPU:0*
T0*
_output_shapes	
:Ш
Read_15/DisableCopyOnReadDisableCopyOnRead+read_15_disablecopyonread_false_negatives_8"/device:CPU:0*
_output_shapes
 Њ
Read_15/ReadVariableOpReadVariableOp+read_15_disablecopyonread_false_negatives_8^Read_15/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:Ш*
dtype0l
Identity_30IdentityRead_15/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:Шb
Identity_31IdentityIdentity_30:output:0"/device:CPU:0*
T0*
_output_shapes	
:Ш
Read_16/DisableCopyOnReadDisableCopyOnRead<read_16_disablecopyonread_training_16_adam_dense_44_kernel_m"/device:CPU:0*
_output_shapes
 О
Read_16/ReadVariableOpReadVariableOp<read_16_disablecopyonread_training_16_adam_dense_44_kernel_m^Read_16/DisableCopyOnRead"/device:CPU:0*
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

: 
Read_17/DisableCopyOnReadDisableCopyOnRead:read_17_disablecopyonread_training_16_adam_dense_44_bias_m"/device:CPU:0*
_output_shapes
 И
Read_17/ReadVariableOpReadVariableOp:read_17_disablecopyonread_training_16_adam_dense_44_bias_m^Read_17/DisableCopyOnRead"/device:CPU:0*
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
:
Read_18/DisableCopyOnReadDisableCopyOnReadBread_18_disablecopyonread_training_16_adam_gru_8_gru_cell_kernel_m"/device:CPU:0*
_output_shapes
 Ф
Read_18/ReadVariableOpReadVariableOpBread_18_disablecopyonread_training_16_adam_gru_8_gru_cell_kernel_m^Read_18/DisableCopyOnRead"/device:CPU:0*
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

:`Ё
Read_19/DisableCopyOnReadDisableCopyOnReadLread_19_disablecopyonread_training_16_adam_gru_8_gru_cell_recurrent_kernel_m"/device:CPU:0*
_output_shapes
 Ю
Read_19/ReadVariableOpReadVariableOpLread_19_disablecopyonread_training_16_adam_gru_8_gru_cell_recurrent_kernel_m^Read_19/DisableCopyOnRead"/device:CPU:0*
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

: `
Read_20/DisableCopyOnReadDisableCopyOnRead@read_20_disablecopyonread_training_16_adam_gru_8_gru_cell_bias_m"/device:CPU:0*
_output_shapes
 Т
Read_20/ReadVariableOpReadVariableOp@read_20_disablecopyonread_training_16_adam_gru_8_gru_cell_bias_m^Read_20/DisableCopyOnRead"/device:CPU:0*
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

:`
Read_21/DisableCopyOnReadDisableCopyOnRead<read_21_disablecopyonread_training_16_adam_dense_44_kernel_v"/device:CPU:0*
_output_shapes
 О
Read_21/ReadVariableOpReadVariableOp<read_21_disablecopyonread_training_16_adam_dense_44_kernel_v^Read_21/DisableCopyOnRead"/device:CPU:0*
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

: 
Read_22/DisableCopyOnReadDisableCopyOnRead:read_22_disablecopyonread_training_16_adam_dense_44_bias_v"/device:CPU:0*
_output_shapes
 И
Read_22/ReadVariableOpReadVariableOp:read_22_disablecopyonread_training_16_adam_dense_44_bias_v^Read_22/DisableCopyOnRead"/device:CPU:0*
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
:
Read_23/DisableCopyOnReadDisableCopyOnReadBread_23_disablecopyonread_training_16_adam_gru_8_gru_cell_kernel_v"/device:CPU:0*
_output_shapes
 Ф
Read_23/ReadVariableOpReadVariableOpBread_23_disablecopyonread_training_16_adam_gru_8_gru_cell_kernel_v^Read_23/DisableCopyOnRead"/device:CPU:0*
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

:`Ё
Read_24/DisableCopyOnReadDisableCopyOnReadLread_24_disablecopyonread_training_16_adam_gru_8_gru_cell_recurrent_kernel_v"/device:CPU:0*
_output_shapes
 Ю
Read_24/ReadVariableOpReadVariableOpLread_24_disablecopyonread_training_16_adam_gru_8_gru_cell_recurrent_kernel_v^Read_24/DisableCopyOnRead"/device:CPU:0*
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

: `
Read_25/DisableCopyOnReadDisableCopyOnRead@read_25_disablecopyonread_training_16_adam_gru_8_gru_cell_bias_v"/device:CPU:0*
_output_shapes
 Т
Read_25/ReadVariableOpReadVariableOp@read_25_disablecopyonread_training_16_adam_gru_8_gru_cell_bias_v^Read_25/DisableCopyOnRead"/device:CPU:0*
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

:`Ѓ
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Ь
valueТBПB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/1/true_positives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/1/true_negatives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/1/false_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/1/false_negatives/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHЃ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*I
value@B>B B B B B B B B B B B B B B B B B B B B B B B B B B B Љ
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Identity_1:output:0Identity_3:output:0Identity_5:output:0Identity_7:output:0Identity_9:output:0Identity_11:output:0Identity_13:output:0Identity_15:output:0Identity_17:output:0Identity_19:output:0Identity_21:output:0Identity_23:output:0Identity_25:output:0Identity_27:output:0Identity_29:output:0Identity_31:output:0Identity_33:output:0Identity_35:output:0Identity_37:output:0Identity_39:output:0Identity_41:output:0Identity_43:output:0Identity_45:output:0Identity_47:output:0Identity_49:output:0Identity_51:output:0savev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *)
dtypes
2	
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:Г
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
: 
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

_user_specified_nameConst:FB
@
_user_specified_name(&training_16/Adam/gru_8/gru_cell/bias/v:RN
L
_user_specified_name42training_16/Adam/gru_8/gru_cell/recurrent_kernel/v:HD
B
_user_specified_name*(training_16/Adam/gru_8/gru_cell/kernel/v:@<
:
_user_specified_name" training_16/Adam/dense_44/bias/v:B>
<
_user_specified_name$"training_16/Adam/dense_44/kernel/v:FB
@
_user_specified_name(&training_16/Adam/gru_8/gru_cell/bias/m:RN
L
_user_specified_name42training_16/Adam/gru_8/gru_cell/recurrent_kernel/m:HD
B
_user_specified_name*(training_16/Adam/gru_8/gru_cell/kernel/m:@<
:
_user_specified_name" training_16/Adam/dense_44/bias/m:B>
<
_user_specified_name$"training_16/Adam/dense_44/kernel/m:1-
+
_user_specified_namefalse_negatives_8:1-
+
_user_specified_namefalse_positives_8:0,
*
_user_specified_nametrue_negatives_8:0,
*
_user_specified_nametrue_positives_8:'#
!
_user_specified_name	count_8:'#
!
_user_specified_name	total_8:>
:
8
_user_specified_name training_16/Adam/learning_rate:6	2
0
_user_specified_nametraining_16/Adam/decay:73
1
_user_specified_nametraining_16/Adam/beta_2:73
1
_user_specified_nametraining_16/Adam/beta_1:51
/
_user_specified_nametraining_16/Adam/iter:3/
-
_user_specified_namegru_8/gru_cell/bias:?;
9
_user_specified_name!gru_8/gru_cell/recurrent_kernel:51
/
_user_specified_namegru_8/gru_cell/kernel:-)
'
_user_specified_namedense_44/bias:/+
)
_user_specified_namedense_44/kernel:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix

к
!__inference__traced_restore_50921
file_prefix2
 assignvariableop_dense_44_kernel: .
 assignvariableop_1_dense_44_bias::
(assignvariableop_2_gru_8_gru_cell_kernel:`D
2assignvariableop_3_gru_8_gru_cell_recurrent_kernel: `8
&assignvariableop_4_gru_8_gru_cell_bias:`2
(assignvariableop_5_training_16_adam_iter:	 4
*assignvariableop_6_training_16_adam_beta_1: 4
*assignvariableop_7_training_16_adam_beta_2: 3
)assignvariableop_8_training_16_adam_decay: ;
1assignvariableop_9_training_16_adam_learning_rate: %
assignvariableop_10_total_8: %
assignvariableop_11_count_8: 3
$assignvariableop_12_true_positives_8:	Ш3
$assignvariableop_13_true_negatives_8:	Ш4
%assignvariableop_14_false_positives_8:	Ш4
%assignvariableop_15_false_negatives_8:	ШH
6assignvariableop_16_training_16_adam_dense_44_kernel_m: B
4assignvariableop_17_training_16_adam_dense_44_bias_m:N
<assignvariableop_18_training_16_adam_gru_8_gru_cell_kernel_m:`X
Fassignvariableop_19_training_16_adam_gru_8_gru_cell_recurrent_kernel_m: `L
:assignvariableop_20_training_16_adam_gru_8_gru_cell_bias_m:`H
6assignvariableop_21_training_16_adam_dense_44_kernel_v: B
4assignvariableop_22_training_16_adam_dense_44_bias_v:N
<assignvariableop_23_training_16_adam_gru_8_gru_cell_kernel_v:`X
Fassignvariableop_24_training_16_adam_gru_8_gru_cell_recurrent_kernel_v: `L
:assignvariableop_25_training_16_adam_gru_8_gru_cell_bias_v:`
identity_27ЂAssignVariableOpЂAssignVariableOp_1ЂAssignVariableOp_10ЂAssignVariableOp_11ЂAssignVariableOp_12ЂAssignVariableOp_13ЂAssignVariableOp_14ЂAssignVariableOp_15ЂAssignVariableOp_16ЂAssignVariableOp_17ЂAssignVariableOp_18ЂAssignVariableOp_19ЂAssignVariableOp_2ЂAssignVariableOp_20ЂAssignVariableOp_21ЂAssignVariableOp_22ЂAssignVariableOp_23ЂAssignVariableOp_24ЂAssignVariableOp_25ЂAssignVariableOp_3ЂAssignVariableOp_4ЂAssignVariableOp_5ЂAssignVariableOp_6ЂAssignVariableOp_7ЂAssignVariableOp_8ЂAssignVariableOp_9І
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Ь
valueТBПB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/1/true_positives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/1/true_negatives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/1/false_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/1/false_negatives/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHІ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*I
value@B>B B B B B B B B B B B B B B B B B B B B B B B B B B B І
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*
_output_shapesn
l:::::::::::::::::::::::::::*)
dtypes
2	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:Г
AssignVariableOpAssignVariableOp assignvariableop_dense_44_kernelIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:З
AssignVariableOp_1AssignVariableOp assignvariableop_1_dense_44_biasIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:П
AssignVariableOp_2AssignVariableOp(assignvariableop_2_gru_8_gru_cell_kernelIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:Щ
AssignVariableOp_3AssignVariableOp2assignvariableop_3_gru_8_gru_cell_recurrent_kernelIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:Н
AssignVariableOp_4AssignVariableOp&assignvariableop_4_gru_8_gru_cell_biasIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0	*
_output_shapes
:П
AssignVariableOp_5AssignVariableOp(assignvariableop_5_training_16_adam_iterIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:С
AssignVariableOp_6AssignVariableOp*assignvariableop_6_training_16_adam_beta_1Identity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:С
AssignVariableOp_7AssignVariableOp*assignvariableop_7_training_16_adam_beta_2Identity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:Р
AssignVariableOp_8AssignVariableOp)assignvariableop_8_training_16_adam_decayIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:Ш
AssignVariableOp_9AssignVariableOp1assignvariableop_9_training_16_adam_learning_rateIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:Д
AssignVariableOp_10AssignVariableOpassignvariableop_10_total_8Identity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:Д
AssignVariableOp_11AssignVariableOpassignvariableop_11_count_8Identity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:Н
AssignVariableOp_12AssignVariableOp$assignvariableop_12_true_positives_8Identity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:Н
AssignVariableOp_13AssignVariableOp$assignvariableop_13_true_negatives_8Identity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:О
AssignVariableOp_14AssignVariableOp%assignvariableop_14_false_positives_8Identity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:О
AssignVariableOp_15AssignVariableOp%assignvariableop_15_false_negatives_8Identity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:Я
AssignVariableOp_16AssignVariableOp6assignvariableop_16_training_16_adam_dense_44_kernel_mIdentity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:Э
AssignVariableOp_17AssignVariableOp4assignvariableop_17_training_16_adam_dense_44_bias_mIdentity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:е
AssignVariableOp_18AssignVariableOp<assignvariableop_18_training_16_adam_gru_8_gru_cell_kernel_mIdentity_18:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:п
AssignVariableOp_19AssignVariableOpFassignvariableop_19_training_16_adam_gru_8_gru_cell_recurrent_kernel_mIdentity_19:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:г
AssignVariableOp_20AssignVariableOp:assignvariableop_20_training_16_adam_gru_8_gru_cell_bias_mIdentity_20:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:Я
AssignVariableOp_21AssignVariableOp6assignvariableop_21_training_16_adam_dense_44_kernel_vIdentity_21:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:Э
AssignVariableOp_22AssignVariableOp4assignvariableop_22_training_16_adam_dense_44_bias_vIdentity_22:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:е
AssignVariableOp_23AssignVariableOp<assignvariableop_23_training_16_adam_gru_8_gru_cell_kernel_vIdentity_23:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:п
AssignVariableOp_24AssignVariableOpFassignvariableop_24_training_16_adam_gru_8_gru_cell_recurrent_kernel_vIdentity_24:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:г
AssignVariableOp_25AssignVariableOp:assignvariableop_25_training_16_adam_gru_8_gru_cell_bias_vIdentity_25:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 
Identity_26Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_27IdentityIdentity_26:output:0^NoOp_1*
T0*
_output_shapes
: д
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
AssignVariableOpAssignVariableOp:FB
@
_user_specified_name(&training_16/Adam/gru_8/gru_cell/bias/v:RN
L
_user_specified_name42training_16/Adam/gru_8/gru_cell/recurrent_kernel/v:HD
B
_user_specified_name*(training_16/Adam/gru_8/gru_cell/kernel/v:@<
:
_user_specified_name" training_16/Adam/dense_44/bias/v:B>
<
_user_specified_name$"training_16/Adam/dense_44/kernel/v:FB
@
_user_specified_name(&training_16/Adam/gru_8/gru_cell/bias/m:RN
L
_user_specified_name42training_16/Adam/gru_8/gru_cell/recurrent_kernel/m:HD
B
_user_specified_name*(training_16/Adam/gru_8/gru_cell/kernel/m:@<
:
_user_specified_name" training_16/Adam/dense_44/bias/m:B>
<
_user_specified_name$"training_16/Adam/dense_44/kernel/m:1-
+
_user_specified_namefalse_negatives_8:1-
+
_user_specified_namefalse_positives_8:0,
*
_user_specified_nametrue_negatives_8:0,
*
_user_specified_nametrue_positives_8:'#
!
_user_specified_name	count_8:'#
!
_user_specified_name	total_8:>
:
8
_user_specified_name training_16/Adam/learning_rate:6	2
0
_user_specified_nametraining_16/Adam/decay:73
1
_user_specified_nametraining_16/Adam/beta_2:73
1
_user_specified_nametraining_16/Adam/beta_1:51
/
_user_specified_nametraining_16/Adam/iter:3/
-
_user_specified_namegru_8/gru_cell/bias:?;
9
_user_specified_name!gru_8/gru_cell/recurrent_kernel:51
/
_user_specified_namegru_8/gru_cell/kernel:-)
'
_user_specified_namedense_44/bias:/+
)
_user_specified_namedense_44/kernel:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
р

C__inference_gru_cell_layer_call_and_return_conditional_losses_50617

inputs
states_04
"readvariableop_gru_8_gru_cell_bias:`=
+matmul_readvariableop_gru_8_gru_cell_kernel:`I
7matmul_1_readvariableop_gru_8_gru_cell_recurrent_kernel: `
identity

identity_1ЂMatMul/ReadVariableOpЂMatMul_1/ReadVariableOpЂReadVariableOpq
ReadVariableOpReadVariableOp"readvariableop_gru_8_gru_cell_bias*
_output_shapes

:`*
dtype0_
unstackUnpackReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
num
MatMul/ReadVariableOpReadVariableOp+matmul_readvariableop_gru_8_gru_cell_kernel*
_output_shapes

:`*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ`h
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџ`Z
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџЃ
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ *
	num_split
MatMul_1/ReadVariableOpReadVariableOp7matmul_1_readvariableop_gru_8_gru_cell_recurrent_kernel*
_output_shapes

: `*
dtype0o
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ`l
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџ`Z
ConstConst*
_output_shapes
:*
dtype0*!
valueB"        џџџџ\
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџЦ
split_1SplitVBiasAdd_1:output:0Const:output:0split_1/split_dim:output:0*

Tlen0*
T0*M
_output_shapes;
9:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ *
	num_split`
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ M
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:џџџџџџџџџ b
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ Q
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ ]
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ Y
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ I
TanhTanh	add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ U
mul_1MulSigmoid:y:0states_0*
T0*'
_output_shapes
:џџџџџџџџџ J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ Q
mul_2Mulsub:z:0Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ V
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ X
IdentityIdentity	add_3:z:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ Z

Identity_1Identity	add_3:z:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ e
NoOpNoOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
_output_shapes
 "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:џџџџџџџџџ:џџџџџџџџџ : : : 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp:?;
9
_user_specified_name!gru_8/gru_cell/recurrent_kernel:51
/
_user_specified_namegru_8/gru_cell/kernel:3/
-
_user_specified_namegru_8/gru_cell/bias:QM
'
_output_shapes
:џџџџџџџџџ 
"
_user_specified_name
states_0:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ВN
 
@__inference_gru_8_layer_call_and_return_conditional_losses_50538

inputs=
+gru_cell_readvariableop_gru_8_gru_cell_bias:`F
4gru_cell_matmul_readvariableop_gru_8_gru_cell_kernel:`R
@gru_cell_matmul_1_readvariableop_gru_8_gru_cell_recurrent_kernel: `
identityЂgru_cell/MatMul/ReadVariableOpЂ gru_cell/MatMul_1/ReadVariableOpЂgru_cell/ReadVariableOpЂwhileI
ShapeShapeinputs*
T0*
_output_shapes
::эЯ]
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
valueB:б
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
:џџџџџџџџџ c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          n
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:ГџџџџџџџџџR
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
::эЯ_
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
valueB:л
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
џџџџџџџџџД
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   р
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв_
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
valueB:щ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask
gru_cell/ReadVariableOpReadVariableOp+gru_cell_readvariableop_gru_8_gru_cell_bias*
_output_shapes

:`*
dtype0q
gru_cell/unstackUnpackgru_cell/ReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
num
gru_cell/MatMul/ReadVariableOpReadVariableOp4gru_cell_matmul_readvariableop_gru_8_gru_cell_kernel*
_output_shapes

:`*
dtype0
gru_cell/MatMulMatMulstrided_slice_2:output:0&gru_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ`
gru_cell/BiasAddBiasAddgru_cell/MatMul:product:0gru_cell/unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџ`c
gru_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџО
gru_cell/splitSplit!gru_cell/split/split_dim:output:0gru_cell/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ *
	num_splitЁ
 gru_cell/MatMul_1/ReadVariableOpReadVariableOp@gru_cell_matmul_1_readvariableop_gru_8_gru_cell_recurrent_kernel*
_output_shapes

: `*
dtype0
gru_cell/MatMul_1MatMulzeros:output:0(gru_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ`
gru_cell/BiasAdd_1BiasAddgru_cell/MatMul_1:product:0gru_cell/unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџ`c
gru_cell/ConstConst*
_output_shapes
:*
dtype0*!
valueB"        џџџџe
gru_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџъ
gru_cell/split_1SplitVgru_cell/BiasAdd_1:output:0gru_cell/Const:output:0#gru_cell/split_1/split_dim:output:0*

Tlen0*
T0*M
_output_shapes;
9:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ *
	num_split{
gru_cell/addAddV2gru_cell/split:output:0gru_cell/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ _
gru_cell/SigmoidSigmoidgru_cell/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ }
gru_cell/add_1AddV2gru_cell/split:output:1gru_cell/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ c
gru_cell/Sigmoid_1Sigmoidgru_cell/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ x
gru_cell/mulMulgru_cell/Sigmoid_1:y:0gru_cell/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ t
gru_cell/add_2AddV2gru_cell/split:output:2gru_cell/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ [
gru_cell/TanhTanhgru_cell/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ m
gru_cell/mul_1Mulgru_cell/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:џџџџџџџџџ S
gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?t
gru_cell/subSubgru_cell/sub/x:output:0gru_cell/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ l
gru_cell/mul_2Mulgru_cell/sub:z:0gru_cell/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ q
gru_cell/add_3AddV2gru_cell/mul_1:z:0gru_cell/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :Х
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшвF
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
џџџџџџџџџT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Х
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+gru_cell_readvariableop_gru_8_gru_cell_bias4gru_cell_matmul_readvariableop_gru_8_gru_cell_kernel@gru_cell_matmul_1_readvariableop_gru_8_gru_cell_recurrent_kernel*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :џџџџџџџџџ : : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_50448*
condR
while_cond_50447*8
output_shapes'
%: : : : :џџџџџџџџџ : : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    ж
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:џџџџџџџџџ *
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ 
NoOpNoOp^gru_cell/MatMul/ReadVariableOp!^gru_cell/MatMul_1/ReadVariableOp^gru_cell/ReadVariableOp^while*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:џџџџџџџџџГ: : : 2@
gru_cell/MatMul/ReadVariableOpgru_cell/MatMul/ReadVariableOp2D
 gru_cell/MatMul_1/ReadVariableOp gru_cell/MatMul_1/ReadVariableOp22
gru_cell/ReadVariableOpgru_cell/ReadVariableOp2
whilewhile:?;
9
_user_specified_name!gru_8/gru_cell/recurrent_kernel:51
/
_user_specified_namegru_8/gru_cell/kernel:3/
-
_user_specified_namegru_8/gru_cell/bias:T P
,
_output_shapes
:џџџџџџџџџГ
 
_user_specified_nameinputs
Н
k
A__inference_dot_17_layer_call_and_return_conditional_losses_49356

inputs
inputs_1
identity`
MatMulBatchMatMulV2inputsinputs_1*
T0*,
_output_shapes
:џџџџџџџџџГ
R
ShapeShapeMatMul:output:0*
T0*
_output_shapes
::эЯ\
IdentityIdentityMatMul:output:0*
T0*,
_output_shapes
:џџџџџџџџџГ
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:џџџџџџџџџГГ:џџџџџџџџџГ
:TP
,
_output_shapes
:џџџџџџџџџГ

 
_user_specified_nameinputs:U Q
-
_output_shapes
:џџџџџџџџџГГ
 
_user_specified_nameinputs
еJ


model_52_gru_8_while_body_48893%
!model_52_gru_8_while_loop_counter+
'model_52_gru_8_while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3$
 model_52_gru_8_strided_slice_1_0`
\tensorarrayv2read_tensorlistgetitem_model_52_gru_8_tensorarrayunstack_tensorlistfromtensor_0d
`tensorarrayv2read_1_tensorlistgetitem_model_52_gru_8_tensorarrayunstack_1_tensorlistfromtensor_0?
-gru_cell_readvariableop_gru_8_gru_cell_bias_0:`H
6gru_cell_matmul_readvariableop_gru_8_gru_cell_kernel_0:`T
Bgru_cell_matmul_1_readvariableop_gru_8_gru_cell_recurrent_kernel_0: `
identity

identity_1

identity_2

identity_3

identity_4

identity_5"
model_52_gru_8_strided_slice_1^
Ztensorarrayv2read_tensorlistgetitem_model_52_gru_8_tensorarrayunstack_tensorlistfromtensorb
^tensorarrayv2read_1_tensorlistgetitem_model_52_gru_8_tensorarrayunstack_1_tensorlistfromtensor=
+gru_cell_readvariableop_gru_8_gru_cell_bias:`F
4gru_cell_matmul_readvariableop_gru_8_gru_cell_kernel:`R
@gru_cell_matmul_1_readvariableop_gru_8_gru_cell_recurrent_kernel: `Ђgru_cell/MatMul/ReadVariableOpЂ gru_cell/MatMul_1/ReadVariableOpЂgru_cell/ReadVariableOp
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   
#TensorArrayV2Read/TensorListGetItemTensorListGetItem\tensorarrayv2read_tensorlistgetitem_model_52_gru_8_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype0
3TensorArrayV2Read_1/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   Ѕ
%TensorArrayV2Read_1/TensorListGetItemTensorListGetItem`tensorarrayv2read_1_tensorlistgetitem_model_52_gru_8_tensorarrayunstack_1_tensorlistfromtensor_0placeholder<TensorArrayV2Read_1/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype0

gru_cell/ReadVariableOpReadVariableOp-gru_cell_readvariableop_gru_8_gru_cell_bias_0*
_output_shapes

:`*
dtype0q
gru_cell/unstackUnpackgru_cell/ReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
num
gru_cell/MatMul/ReadVariableOpReadVariableOp6gru_cell_matmul_readvariableop_gru_8_gru_cell_kernel_0*
_output_shapes

:`*
dtype0
gru_cell/MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0&gru_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ`
gru_cell/BiasAddBiasAddgru_cell/MatMul:product:0gru_cell/unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџ`c
gru_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџО
gru_cell/splitSplit!gru_cell/split/split_dim:output:0gru_cell/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ *
	num_splitЃ
 gru_cell/MatMul_1/ReadVariableOpReadVariableOpBgru_cell_matmul_1_readvariableop_gru_8_gru_cell_recurrent_kernel_0*
_output_shapes

: `*
dtype0
gru_cell/MatMul_1MatMulplaceholder_3(gru_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ`
gru_cell/BiasAdd_1BiasAddgru_cell/MatMul_1:product:0gru_cell/unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџ`c
gru_cell/ConstConst*
_output_shapes
:*
dtype0*!
valueB"        џџџџe
gru_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџъ
gru_cell/split_1SplitVgru_cell/BiasAdd_1:output:0gru_cell/Const:output:0#gru_cell/split_1/split_dim:output:0*

Tlen0*
T0*M
_output_shapes;
9:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ *
	num_split{
gru_cell/addAddV2gru_cell/split:output:0gru_cell/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ _
gru_cell/SigmoidSigmoidgru_cell/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ }
gru_cell/add_1AddV2gru_cell/split:output:1gru_cell/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ c
gru_cell/Sigmoid_1Sigmoidgru_cell/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ x
gru_cell/mulMulgru_cell/Sigmoid_1:y:0gru_cell/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ t
gru_cell/add_2AddV2gru_cell/split:output:2gru_cell/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ [
gru_cell/TanhTanhgru_cell/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ l
gru_cell/mul_1Mulgru_cell/Sigmoid:y:0placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ S
gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?t
gru_cell/subSubgru_cell/sub/x:output:0gru_cell/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ l
gru_cell/mul_2Mulgru_cell/sub:z:0gru_cell/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ q
gru_cell/add_3AddV2gru_cell/mul_1:z:0gru_cell/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ _
Tile/multiplesConst*
_output_shapes
:*
dtype0*
valueB"      
TileTile,TensorArrayV2Read_1/TensorListGetItem:item:0Tile/multiples:output:0*
T0
*'
_output_shapes
:џџџџџџџџџx
SelectV2SelectV2Tile:output:0gru_cell/add_3:z:0placeholder_2*
T0*'
_output_shapes
:џџџџџџџџџ a
Tile_1/multiplesConst*
_output_shapes
:*
dtype0*
valueB"      
Tile_1Tile,TensorArrayV2Read_1/TensorListGetItem:item:0Tile_1/multiples:output:0*
T0
*'
_output_shapes
:џџџџџџџџџ|

SelectV2_1SelectV2Tile_1:output:0gru_cell/add_3:z:0placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ l
*TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : а
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_13TensorArrayV2Write/TensorListSetItem/index:output:0SelectV2:output:0*
_output_shapes
: *
element_dtype0:щшвG
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
value	B :d
add_1AddV2!model_52_gru_8_while_loop_counteradd_1/y:output:0*
T0*
_output_shapes
: G
IdentityIdentity	add_1:z:0^NoOp*
T0*
_output_shapes
: g

Identity_1Identity'model_52_gru_8_while_maximum_iterations^NoOp*
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
:џџџџџџџџџ d

Identity_5IdentitySelectV2_1:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ 
NoOpNoOp^gru_cell/MatMul/ReadVariableOp!^gru_cell/MatMul_1/ReadVariableOp^gru_cell/ReadVariableOp*
_output_shapes
 "
@gru_cell_matmul_1_readvariableop_gru_8_gru_cell_recurrent_kernelBgru_cell_matmul_1_readvariableop_gru_8_gru_cell_recurrent_kernel_0"n
4gru_cell_matmul_readvariableop_gru_8_gru_cell_kernel6gru_cell_matmul_readvariableop_gru_8_gru_cell_kernel_0"\
+gru_cell_readvariableop_gru_8_gru_cell_bias-gru_cell_readvariableop_gru_8_gru_cell_bias_0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"
identityIdentity:output:0"B
model_52_gru_8_strided_slice_1 model_52_gru_8_strided_slice_1_0"Т
^tensorarrayv2read_1_tensorlistgetitem_model_52_gru_8_tensorarrayunstack_1_tensorlistfromtensor`tensorarrayv2read_1_tensorlistgetitem_model_52_gru_8_tensorarrayunstack_1_tensorlistfromtensor_0"К
Ztensorarrayv2read_tensorlistgetitem_model_52_gru_8_tensorarrayunstack_tensorlistfromtensor\tensorarrayv2read_tensorlistgetitem_model_52_gru_8_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :џџџџџџџџџ :џџџџџџџџџ : : : : : : 2@
gru_cell/MatMul/ReadVariableOpgru_cell/MatMul/ReadVariableOp2D
 gru_cell/MatMul_1/ReadVariableOp gru_cell/MatMul_1/ReadVariableOp22
gru_cell/ReadVariableOpgru_cell/ReadVariableOp:?;
9
_user_specified_name!gru_8/gru_cell/recurrent_kernel:5
1
/
_user_specified_namegru_8/gru_cell/kernel:3	/
-
_user_specified_namegru_8/gru_cell/bias:pl

_output_shapes
: 
R
_user_specified_name:8model_52/gru_8/TensorArrayUnstack_1/TensorListFromTensor:nj

_output_shapes
: 
P
_user_specified_name86model_52/gru_8/TensorArrayUnstack/TensorListFromTensor:VR

_output_shapes
: 
8
_user_specified_name model_52/gru_8/strided_slice_1:-)
'
_output_shapes
:џџџџџџџџџ :-)
'
_output_shapes
:џџџџџџџџџ :

_output_shapes
: :

_output_shapes
: :_[

_output_shapes
: 
A
_user_specified_name)'model_52/gru_8/while/maximum_iterations:Y U

_output_shapes
: 
;
_user_specified_name#!model_52/gru_8/while/loop_counter
ш
c
G__inference_activation_8_layer_call_and_return_conditional_losses_49348

inputs
identityR
SoftmaxSoftmaxinputs*
T0*-
_output_shapes
:џџџџџџџџџГГ_
IdentityIdentitySoftmax:softmax:0*
T0*-
_output_shapes
:џџџџџџџџџГГ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:џџџџџџџџџГГ:U Q
-
_output_shapes
:џџџџџџџџџГГ
 
_user_specified_nameinputs
і
у
while_cond_50292
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
less_strided_slice_1-
)while_cond_50292___redundant_placeholder0-
)while_cond_50292___redundant_placeholder1-
)while_cond_50292___redundant_placeholder2-
)while_cond_50292___redundant_placeholder3
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
-: : : : :џџџџџџџџџ : :::::
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
:џџџџџџџџџ :
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
с

Ќ
model_52_gru_8_while_cond_48892%
!model_52_gru_8_while_loop_counter+
'model_52_gru_8_while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3'
#less_model_52_gru_8_strided_slice_1<
8model_52_gru_8_while_cond_48892___redundant_placeholder0<
8model_52_gru_8_while_cond_48892___redundant_placeholder1<
8model_52_gru_8_while_cond_48892___redundant_placeholder2<
8model_52_gru_8_while_cond_48892___redundant_placeholder3<
8model_52_gru_8_while_cond_48892___redundant_placeholder4
identity
_
LessLessplaceholder#less_model_52_gru_8_strided_slice_1*
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
D: : : : :џџџџџџџџџ :џџџџџџџџџ : ::::::

_output_shapes
::

_output_shapes
::VR

_output_shapes
: 
8
_user_specified_name model_52/gru_8/strided_slice_1:-)
'
_output_shapes
:џџџџџџџџџ :-)
'
_output_shapes
:џџџџџџџџџ :

_output_shapes
: :

_output_shapes
: :_[

_output_shapes
: 
A
_user_specified_name)'model_52/gru_8/while/maximum_iterations:Y U

_output_shapes
: 
;
_user_specified_name#!model_52/gru_8/while/loop_counter
ю6
Ј
@__inference_gru_8_layer_call_and_return_conditional_losses_49281

inputs.
gru_cell_gru_8_gru_cell_bias:`0
gru_cell_gru_8_gru_cell_kernel:`:
(gru_cell_gru_8_gru_cell_recurrent_kernel: `
identityЂ gru_cell/StatefulPartitionedCallЂwhileI
ShapeShapeinputs*
T0*
_output_shapes
::эЯ]
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
valueB:б
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
:џџџџџџџџџ c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџR
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
::эЯ_
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
valueB:л
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
џџџџџџџџџД
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   р
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв_
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
valueB:щ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_maskю
 gru_cell/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0gru_cell_gru_8_gru_cell_biasgru_cell_gru_8_gru_cell_kernel(gru_cell_gru_8_gru_cell_recurrent_kernel*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:џџџџџџџџџ :џџџџџџџџџ *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_gru_cell_layer_call_and_return_conditional_losses_49208n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :Х
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшвF
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
џџџџџџџџџT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0gru_cell_gru_8_gru_cell_biasgru_cell_gru_8_gru_cell_kernel(gru_cell_gru_8_gru_cell_recurrent_kernel*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :џџџџџџџџџ : : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_49219*
condR
while_cond_49218*8
output_shapes'
%: : : : :џџџџџџџџџ : : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    ж
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:џџџџџџџџџ *
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ M
NoOpNoOp!^gru_cell/StatefulPartitionedCall^while*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:џџџџџџџџџџџџџџџџџџ: : : 2D
 gru_cell/StatefulPartitionedCall gru_cell/StatefulPartitionedCall2
whilewhile:?;
9
_user_specified_name!gru_8/gru_cell/recurrent_kernel:51
/
_user_specified_namegru_8/gru_cell/kernel:3/
-
_user_specified_namegru_8/gru_cell/bias:\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
і
у
while_cond_49218
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
less_strided_slice_1-
)while_cond_49218___redundant_placeholder0-
)while_cond_49218___redundant_placeholder1-
)while_cond_49218___redundant_placeholder2-
)while_cond_49218___redundant_placeholder3
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
-: : : : :џџџџџџџџџ : :::::
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
:џџџџџџџџџ :
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

Ё
(__inference_dense_44_layer_call_fn_50545

inputs!
dense_44_kernel: 
dense_44_bias:
identityЂStatefulPartitionedCallф
StatefulPartitionedCallStatefulPartitionedCallinputsdense_44_kerneldense_44_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_44_layer_call_and_return_conditional_losses_49542o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ : : 22
StatefulPartitionedCallStatefulPartitionedCall:-)
'
_user_specified_namedense_44/bias:/+
)
_user_specified_namedense_44/kernel:O K
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
ВN
 
@__inference_gru_8_layer_call_and_return_conditional_losses_49710

inputs=
+gru_cell_readvariableop_gru_8_gru_cell_bias:`F
4gru_cell_matmul_readvariableop_gru_8_gru_cell_kernel:`R
@gru_cell_matmul_1_readvariableop_gru_8_gru_cell_recurrent_kernel: `
identityЂgru_cell/MatMul/ReadVariableOpЂ gru_cell/MatMul_1/ReadVariableOpЂgru_cell/ReadVariableOpЂwhileI
ShapeShapeinputs*
T0*
_output_shapes
::эЯ]
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
valueB:б
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
:џџџџџџџџџ c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          n
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:ГџџџџџџџџџR
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
::эЯ_
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
valueB:л
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
џџџџџџџџџД
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   р
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв_
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
valueB:щ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask
gru_cell/ReadVariableOpReadVariableOp+gru_cell_readvariableop_gru_8_gru_cell_bias*
_output_shapes

:`*
dtype0q
gru_cell/unstackUnpackgru_cell/ReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
num
gru_cell/MatMul/ReadVariableOpReadVariableOp4gru_cell_matmul_readvariableop_gru_8_gru_cell_kernel*
_output_shapes

:`*
dtype0
gru_cell/MatMulMatMulstrided_slice_2:output:0&gru_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ`
gru_cell/BiasAddBiasAddgru_cell/MatMul:product:0gru_cell/unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџ`c
gru_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџО
gru_cell/splitSplit!gru_cell/split/split_dim:output:0gru_cell/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ *
	num_splitЁ
 gru_cell/MatMul_1/ReadVariableOpReadVariableOp@gru_cell_matmul_1_readvariableop_gru_8_gru_cell_recurrent_kernel*
_output_shapes

: `*
dtype0
gru_cell/MatMul_1MatMulzeros:output:0(gru_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ`
gru_cell/BiasAdd_1BiasAddgru_cell/MatMul_1:product:0gru_cell/unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџ`c
gru_cell/ConstConst*
_output_shapes
:*
dtype0*!
valueB"        џџџџe
gru_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџъ
gru_cell/split_1SplitVgru_cell/BiasAdd_1:output:0gru_cell/Const:output:0#gru_cell/split_1/split_dim:output:0*

Tlen0*
T0*M
_output_shapes;
9:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ *
	num_split{
gru_cell/addAddV2gru_cell/split:output:0gru_cell/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ _
gru_cell/SigmoidSigmoidgru_cell/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ }
gru_cell/add_1AddV2gru_cell/split:output:1gru_cell/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ c
gru_cell/Sigmoid_1Sigmoidgru_cell/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ x
gru_cell/mulMulgru_cell/Sigmoid_1:y:0gru_cell/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ t
gru_cell/add_2AddV2gru_cell/split:output:2gru_cell/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ [
gru_cell/TanhTanhgru_cell/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ m
gru_cell/mul_1Mulgru_cell/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:џџџџџџџџџ S
gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?t
gru_cell/subSubgru_cell/sub/x:output:0gru_cell/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ l
gru_cell/mul_2Mulgru_cell/sub:z:0gru_cell/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ q
gru_cell/add_3AddV2gru_cell/mul_1:z:0gru_cell/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :Х
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшвF
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
џџџџџџџџџT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Х
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+gru_cell_readvariableop_gru_8_gru_cell_bias4gru_cell_matmul_readvariableop_gru_8_gru_cell_kernel@gru_cell_matmul_1_readvariableop_gru_8_gru_cell_recurrent_kernel*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :џџџџџџџџџ : : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_49620*
condR
while_cond_49619*8
output_shapes'
%: : : : :џџџџџџџџџ : : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    ж
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:џџџџџџџџџ *
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ 
NoOpNoOp^gru_cell/MatMul/ReadVariableOp!^gru_cell/MatMul_1/ReadVariableOp^gru_cell/ReadVariableOp^while*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:џџџџџџџџџГ: : : 2@
gru_cell/MatMul/ReadVariableOpgru_cell/MatMul/ReadVariableOp2D
 gru_cell/MatMul_1/ReadVariableOp gru_cell/MatMul_1/ReadVariableOp22
gru_cell/ReadVariableOpgru_cell/ReadVariableOp2
whilewhile:?;
9
_user_specified_name!gru_8/gru_cell/recurrent_kernel:51
/
_user_specified_namegru_8/gru_cell/kernel:3/
-
_user_specified_namegru_8/gru_cell/bias:T P
,
_output_shapes
:џџџџџџџџџГ
 
_user_specified_nameinputs
ш
c
G__inference_activation_8_layer_call_and_return_conditional_losses_49848

inputs
identityR
SoftmaxSoftmaxinputs*
T0*-
_output_shapes
:џџџџџџџџџГГ_
IdentityIdentitySoftmax:softmax:0*
T0*-
_output_shapes
:џџџџџџџџџГГ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:џџџџџџџџџГГ:U Q
-
_output_shapes
:џџџџџџџџџГГ
 
_user_specified_nameinputs

k
A__inference_dot_16_layer_call_and_return_conditional_losses_49342

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
:џџџџџџџџџ
Гf
MatMulBatchMatMulV2inputstranspose:y:0*
T0*-
_output_shapes
:џџџџџџџџџГГR
ShapeShapeMatMul:output:0*
T0*
_output_shapes
::эЯ]
IdentityIdentityMatMul:output:0*
T0*-
_output_shapes
:џџџџџџџџџГГ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:џџџџџџџџџГ
:џџџџџџџџџГ
:TP
,
_output_shapes
:џџџџџџџџџГ

 
_user_specified_nameinputs:T P
,
_output_shapes
:џџџџџџџџџГ

 
_user_specified_nameinputs

д
 __inference__wrapped_model_49005
input_priorL
:model_52_gru_8_gru_cell_readvariableop_gru_8_gru_cell_bias:`U
Cmodel_52_gru_8_gru_cell_matmul_readvariableop_gru_8_gru_cell_kernel:`a
Omodel_52_gru_8_gru_cell_matmul_1_readvariableop_gru_8_gru_cell_recurrent_kernel: `I
7model_52_dense_44_matmul_readvariableop_dense_44_kernel: D
6model_52_dense_44_biasadd_readvariableop_dense_44_bias:
identityЂ(model_52/dense_44/BiasAdd/ReadVariableOpЂ'model_52/dense_44/MatMul/ReadVariableOpЂ-model_52/gru_8/gru_cell/MatMul/ReadVariableOpЂ/model_52/gru_8/gru_cell/MatMul_1/ReadVariableOpЂ&model_52/gru_8/gru_cell/ReadVariableOpЂmodel_52/gru_8/whilef
!model_52/masking_prior/NotEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  П
model_52/masking_prior/NotEqualNotEqualinput_prior*model_52/masking_prior/NotEqual/y:output:0*
T0*,
_output_shapes
:џџџџџџџџџГ
w
,model_52/masking_prior/Any/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџМ
model_52/masking_prior/AnyAny#model_52/masking_prior/NotEqual:z:05model_52/masking_prior/Any/reduction_indices:output:0*,
_output_shapes
:џџџџџџџџџГ*
	keep_dims(
model_52/masking_prior/CastCast#model_52/masking_prior/Any:output:0*

DstT0*

SrcT0
*,
_output_shapes
:џџџџџџџџџГ
model_52/masking_prior/mulMulinput_priormodel_52/masking_prior/Cast:y:0*
T0*,
_output_shapes
:џџџџџџџџџГ
Ё
model_52/masking_prior/SqueezeSqueeze#model_52/masking_prior/Any:output:0*
T0
*(
_output_shapes
:џџџџџџџџџГ*
squeeze_dims

џџџџџџџџџs
model_52/dot_16/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          І
model_52/dot_16/transpose	Transposemodel_52/masking_prior/mul:z:0'model_52/dot_16/transpose/perm:output:0*
T0*,
_output_shapes
:џџџџџџџџџ
Г
model_52/dot_16/MatMulBatchMatMulV2model_52/masking_prior/mul:z:0model_52/dot_16/transpose:y:0*
T0*-
_output_shapes
:џџџџџџџџџГГr
model_52/dot_16/ShapeShapemodel_52/dot_16/MatMul:output:0*
T0*
_output_shapes
::эЯ
model_52/activation_8/SoftmaxSoftmaxmodel_52/dot_16/MatMul:output:0*
T0*-
_output_shapes
:џџџџџџџџџГГЇ
model_52/dot_17/MatMulBatchMatMulV2'model_52/activation_8/Softmax:softmax:0model_52/masking_prior/mul:z:0*
T0*,
_output_shapes
:џџџџџџџџџГ
r
model_52/dot_17/ShapeShapemodel_52/dot_17/MatMul:output:0*
T0*
_output_shapes
::эЯ
model_52/multiply_8/mulMulmodel_52/dot_17/MatMul:output:0model_52/masking_prior/mul:z:0*
T0*,
_output_shapes
:џџџџџџџџџГ
d
"model_52/multiply_8/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : Й
model_52/multiply_8/ExpandDims
ExpandDims'model_52/masking_prior/Squeeze:output:0+model_52/multiply_8/ExpandDims/dim:output:0*
T0
*,
_output_shapes
:џџџџџџџџџГg
%model_52/multiply_8/concat/concat_dimConst*
_output_shapes
: *
dtype0*
value	B : 
!model_52/multiply_8/concat/concatIdentity'model_52/multiply_8/ExpandDims:output:0*
T0
*,
_output_shapes
:џџџџџџџџџГk
)model_52/multiply_8/All/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : Ј
model_52/multiply_8/AllAll*model_52/multiply_8/concat/concat:output:02model_52/multiply_8/All/reduction_indices:output:0*(
_output_shapes
:џџџџџџџџџГd
"model_52/concatenate_8/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :г
model_52/concatenate_8/concatConcatV2model_52/masking_prior/mul:z:0model_52/multiply_8/mul:z:0+model_52/concatenate_8/concat/axis:output:0*
N*
T0*,
_output_shapes
:џџџџџџџџџГp
%model_52/concatenate_8/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџП
!model_52/concatenate_8/ExpandDims
ExpandDims'model_52/masking_prior/Squeeze:output:0.model_52/concatenate_8/ExpandDims/dim:output:0*
T0
*,
_output_shapes
:џџџџџџџџџГr
'model_52/concatenate_8/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџМ
#model_52/concatenate_8/ExpandDims_1
ExpandDims model_52/multiply_8/All:output:00model_52/concatenate_8/ExpandDims_1/dim:output:0*
T0
*,
_output_shapes
:џџџџџџџџџГf
$model_52/concatenate_8/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B :є
model_52/concatenate_8/concat_1ConcatV2*model_52/concatenate_8/ExpandDims:output:0,model_52/concatenate_8/ExpandDims_1:output:0-model_52/concatenate_8/concat_1/axis:output:0*
N*
T0
*,
_output_shapes
:џџџџџџџџџГw
,model_52/concatenate_8/All/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџЌ
model_52/concatenate_8/AllAll(model_52/concatenate_8/concat_1:output:05model_52/concatenate_8/All/reduction_indices:output:0*(
_output_shapes
:џџџџџџџџџГx
model_52/gru_8/ShapeShape&model_52/concatenate_8/concat:output:0*
T0*
_output_shapes
::эЯl
"model_52/gru_8/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: n
$model_52/gru_8/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:n
$model_52/gru_8/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
model_52/gru_8/strided_sliceStridedSlicemodel_52/gru_8/Shape:output:0+model_52/gru_8/strided_slice/stack:output:0-model_52/gru_8/strided_slice/stack_1:output:0-model_52/gru_8/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
model_52/gru_8/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :  
model_52/gru_8/zeros/packedPack%model_52/gru_8/strided_slice:output:0&model_52/gru_8/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:_
model_52/gru_8/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
model_52/gru_8/zerosFill$model_52/gru_8/zeros/packed:output:0#model_52/gru_8/zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ r
model_52/gru_8/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ќ
model_52/gru_8/transpose	Transpose&model_52/concatenate_8/concat:output:0&model_52/gru_8/transpose/perm:output:0*
T0*,
_output_shapes
:Гџџџџџџџџџp
model_52/gru_8/Shape_1Shapemodel_52/gru_8/transpose:y:0*
T0*
_output_shapes
::эЯn
$model_52/gru_8/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: p
&model_52/gru_8/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:p
&model_52/gru_8/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:І
model_52/gru_8/strided_slice_1StridedSlicemodel_52/gru_8/Shape_1:output:0-model_52/gru_8/strided_slice_1/stack:output:0/model_52/gru_8/strided_slice_1/stack_1:output:0/model_52/gru_8/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskh
model_52/gru_8/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџЋ
model_52/gru_8/ExpandDims
ExpandDims#model_52/concatenate_8/All:output:0&model_52/gru_8/ExpandDims/dim:output:0*
T0
*,
_output_shapes
:џџџџџџџџџГt
model_52/gru_8/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ќ
model_52/gru_8/transpose_1	Transpose"model_52/gru_8/ExpandDims:output:0(model_52/gru_8/transpose_1/perm:output:0*
T0
*,
_output_shapes
:Гџџџџџџџџџu
*model_52/gru_8/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџс
model_52/gru_8/TensorArrayV2TensorListReserve3model_52/gru_8/TensorArrayV2/element_shape:output:0'model_52/gru_8/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв
Dmodel_52/gru_8/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   
6model_52/gru_8/TensorArrayUnstack/TensorListFromTensorTensorListFromTensormodel_52/gru_8/transpose:y:0Mmodel_52/gru_8/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшвn
$model_52/gru_8/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: p
&model_52/gru_8/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:p
&model_52/gru_8/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Д
model_52/gru_8/strided_slice_2StridedSlicemodel_52/gru_8/transpose:y:0-model_52/gru_8/strided_slice_2/stack:output:0/model_52/gru_8/strided_slice_2/stack_1:output:0/model_52/gru_8/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_maskЁ
&model_52/gru_8/gru_cell/ReadVariableOpReadVariableOp:model_52_gru_8_gru_cell_readvariableop_gru_8_gru_cell_bias*
_output_shapes

:`*
dtype0
model_52/gru_8/gru_cell/unstackUnpack.model_52/gru_8/gru_cell/ReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
numБ
-model_52/gru_8/gru_cell/MatMul/ReadVariableOpReadVariableOpCmodel_52_gru_8_gru_cell_matmul_readvariableop_gru_8_gru_cell_kernel*
_output_shapes

:`*
dtype0К
model_52/gru_8/gru_cell/MatMulMatMul'model_52/gru_8/strided_slice_2:output:05model_52/gru_8/gru_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ`А
model_52/gru_8/gru_cell/BiasAddBiasAdd(model_52/gru_8/gru_cell/MatMul:product:0(model_52/gru_8/gru_cell/unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџ`r
'model_52/gru_8/gru_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџы
model_52/gru_8/gru_cell/splitSplit0model_52/gru_8/gru_cell/split/split_dim:output:0(model_52/gru_8/gru_cell/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ *
	num_splitП
/model_52/gru_8/gru_cell/MatMul_1/ReadVariableOpReadVariableOpOmodel_52_gru_8_gru_cell_matmul_1_readvariableop_gru_8_gru_cell_recurrent_kernel*
_output_shapes

: `*
dtype0Д
 model_52/gru_8/gru_cell/MatMul_1MatMulmodel_52/gru_8/zeros:output:07model_52/gru_8/gru_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ`Д
!model_52/gru_8/gru_cell/BiasAdd_1BiasAdd*model_52/gru_8/gru_cell/MatMul_1:product:0(model_52/gru_8/gru_cell/unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџ`r
model_52/gru_8/gru_cell/ConstConst*
_output_shapes
:*
dtype0*!
valueB"        џџџџt
)model_52/gru_8/gru_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџІ
model_52/gru_8/gru_cell/split_1SplitV*model_52/gru_8/gru_cell/BiasAdd_1:output:0&model_52/gru_8/gru_cell/Const:output:02model_52/gru_8/gru_cell/split_1/split_dim:output:0*

Tlen0*
T0*M
_output_shapes;
9:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ *
	num_splitЈ
model_52/gru_8/gru_cell/addAddV2&model_52/gru_8/gru_cell/split:output:0(model_52/gru_8/gru_cell/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ }
model_52/gru_8/gru_cell/SigmoidSigmoidmodel_52/gru_8/gru_cell/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ Њ
model_52/gru_8/gru_cell/add_1AddV2&model_52/gru_8/gru_cell/split:output:1(model_52/gru_8/gru_cell/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 
!model_52/gru_8/gru_cell/Sigmoid_1Sigmoid!model_52/gru_8/gru_cell/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ Ѕ
model_52/gru_8/gru_cell/mulMul%model_52/gru_8/gru_cell/Sigmoid_1:y:0(model_52/gru_8/gru_cell/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ Ё
model_52/gru_8/gru_cell/add_2AddV2&model_52/gru_8/gru_cell/split:output:2model_52/gru_8/gru_cell/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ y
model_52/gru_8/gru_cell/TanhTanh!model_52/gru_8/gru_cell/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 
model_52/gru_8/gru_cell/mul_1Mul#model_52/gru_8/gru_cell/Sigmoid:y:0model_52/gru_8/zeros:output:0*
T0*'
_output_shapes
:џџџџџџџџџ b
model_52/gru_8/gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Ё
model_52/gru_8/gru_cell/subSub&model_52/gru_8/gru_cell/sub/x:output:0#model_52/gru_8/gru_cell/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 
model_52/gru_8/gru_cell/mul_2Mulmodel_52/gru_8/gru_cell/sub:z:0 model_52/gru_8/gru_cell/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 
model_52/gru_8/gru_cell/add_3AddV2!model_52/gru_8/gru_cell/mul_1:z:0!model_52/gru_8/gru_cell/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ }
,model_52/gru_8/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    m
+model_52/gru_8/TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :ђ
model_52/gru_8/TensorArrayV2_1TensorListReserve5model_52/gru_8/TensorArrayV2_1/element_shape:output:04model_52/gru_8/TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшвU
model_52/gru_8/timeConst*
_output_shapes
: *
dtype0*
value	B : w
,model_52/gru_8/TensorArrayV2_2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџх
model_52/gru_8/TensorArrayV2_2TensorListReserve5model_52/gru_8/TensorArrayV2_2/element_shape:output:0'model_52/gru_8/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0
*

shape_type0:щшШ
Fmodel_52/gru_8/TensorArrayUnstack_1/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   
8model_52/gru_8/TensorArrayUnstack_1/TensorListFromTensorTensorListFromTensormodel_52/gru_8/transpose_1:y:0Omodel_52/gru_8/TensorArrayUnstack_1/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0
*

shape_type0:щшШ{
model_52/gru_8/zeros_like	ZerosLike!model_52/gru_8/gru_cell/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ r
'model_52/gru_8/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџc
!model_52/gru_8/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
model_52/gru_8/whileWhile*model_52/gru_8/while/loop_counter:output:00model_52/gru_8/while/maximum_iterations:output:0model_52/gru_8/time:output:0'model_52/gru_8/TensorArrayV2_1:handle:0model_52/gru_8/zeros_like:y:0model_52/gru_8/zeros:output:0'model_52/gru_8/strided_slice_1:output:0Fmodel_52/gru_8/TensorArrayUnstack/TensorListFromTensor:output_handle:0Hmodel_52/gru_8/TensorArrayUnstack_1/TensorListFromTensor:output_handle:0:model_52_gru_8_gru_cell_readvariableop_gru_8_gru_cell_biasCmodel_52_gru_8_gru_cell_matmul_readvariableop_gru_8_gru_cell_kernelOmodel_52_gru_8_gru_cell_matmul_1_readvariableop_gru_8_gru_cell_recurrent_kernel*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :џџџџџџџџџ :џџџџџџџџџ : : : : : : *%
_read_only_resource_inputs
	
*+
body#R!
model_52_gru_8_while_body_48893*+
cond#R!
model_52_gru_8_while_cond_48892*M
output_shapes<
:: : : : :џџџџџџџџџ :џџџџџџџџџ : : : : : : *
parallel_iterations 
?model_52/gru_8/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    
1model_52/gru_8/TensorArrayV2Stack/TensorListStackTensorListStackmodel_52/gru_8/while:output:3Hmodel_52/gru_8/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:џџџџџџџџџ *
element_dtype0*
num_elementsw
$model_52/gru_8/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџp
&model_52/gru_8/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: p
&model_52/gru_8/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:в
model_52/gru_8/strided_slice_3StridedSlice:model_52/gru_8/TensorArrayV2Stack/TensorListStack:tensor:0-model_52/gru_8/strided_slice_3/stack:output:0/model_52/gru_8/strided_slice_3/stack_1:output:0/model_52/gru_8/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ *
shrink_axis_maskt
model_52/gru_8/transpose_2/permConst*
_output_shapes
:*
dtype0*!
valueB"          У
model_52/gru_8/transpose_2	Transpose:model_52/gru_8/TensorArrayV2Stack/TensorListStack:tensor:0(model_52/gru_8/transpose_2/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ j
model_52/gru_8/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    
'model_52/dense_44/MatMul/ReadVariableOpReadVariableOp7model_52_dense_44_matmul_readvariableop_dense_44_kernel*
_output_shapes

: *
dtype0Ў
model_52/dense_44/MatMulMatMul'model_52/gru_8/strided_slice_3:output:0/model_52/dense_44/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
(model_52/dense_44/BiasAdd/ReadVariableOpReadVariableOp6model_52_dense_44_biasadd_readvariableop_dense_44_bias*
_output_shapes
:*
dtype0Ќ
model_52/dense_44/BiasAddBiasAdd"model_52/dense_44/MatMul:product:00model_52/dense_44/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџz
model_52/dense_44/SoftmaxSoftmax"model_52/dense_44/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџr
IdentityIdentity#model_52/dense_44/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
NoOpNoOp)^model_52/dense_44/BiasAdd/ReadVariableOp(^model_52/dense_44/MatMul/ReadVariableOp.^model_52/gru_8/gru_cell/MatMul/ReadVariableOp0^model_52/gru_8/gru_cell/MatMul_1/ReadVariableOp'^model_52/gru_8/gru_cell/ReadVariableOp^model_52/gru_8/while*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":џџџџџџџџџГ
: : : : : 2T
(model_52/dense_44/BiasAdd/ReadVariableOp(model_52/dense_44/BiasAdd/ReadVariableOp2R
'model_52/dense_44/MatMul/ReadVariableOp'model_52/dense_44/MatMul/ReadVariableOp2^
-model_52/gru_8/gru_cell/MatMul/ReadVariableOp-model_52/gru_8/gru_cell/MatMul/ReadVariableOp2b
/model_52/gru_8/gru_cell/MatMul_1/ReadVariableOp/model_52/gru_8/gru_cell/MatMul_1/ReadVariableOp2P
&model_52/gru_8/gru_cell/ReadVariableOp&model_52/gru_8/gru_cell/ReadVariableOp2,
model_52/gru_8/whilemodel_52/gru_8/while:-)
'
_user_specified_namedense_44/bias:/+
)
_user_specified_namedense_44/kernel:?;
9
_user_specified_name!gru_8/gru_cell/recurrent_kernel:51
/
_user_specified_namegru_8/gru_cell/kernel:3/
-
_user_specified_namegru_8/gru_cell/bias:Y U
,
_output_shapes
:џџџџџџџџџГ

%
_user_specified_nameinput_prior
и

C__inference_gru_cell_layer_call_and_return_conditional_losses_49070

inputs

states4
"readvariableop_gru_8_gru_cell_bias:`=
+matmul_readvariableop_gru_8_gru_cell_kernel:`I
7matmul_1_readvariableop_gru_8_gru_cell_recurrent_kernel: `
identity

identity_1ЂMatMul/ReadVariableOpЂMatMul_1/ReadVariableOpЂReadVariableOpq
ReadVariableOpReadVariableOp"readvariableop_gru_8_gru_cell_bias*
_output_shapes

:`*
dtype0_
unstackUnpackReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
num
MatMul/ReadVariableOpReadVariableOp+matmul_readvariableop_gru_8_gru_cell_kernel*
_output_shapes

:`*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ`h
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџ`Z
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџЃ
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ *
	num_split
MatMul_1/ReadVariableOpReadVariableOp7matmul_1_readvariableop_gru_8_gru_cell_recurrent_kernel*
_output_shapes

: `*
dtype0m
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ`l
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџ`Z
ConstConst*
_output_shapes
:*
dtype0*!
valueB"        џџџџ\
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџЦ
split_1SplitVBiasAdd_1:output:0Const:output:0split_1/split_dim:output:0*

Tlen0*
T0*M
_output_shapes;
9:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ *
	num_split`
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ M
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:џџџџџџџџџ b
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ Q
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ ]
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ Y
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ I
TanhTanh	add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ S
mul_1MulSigmoid:y:0states*
T0*'
_output_shapes
:џџџџџџџџџ J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ Q
mul_2Mulsub:z:0Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ V
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ X
IdentityIdentity	add_3:z:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ Z

Identity_1Identity	add_3:z:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ e
NoOpNoOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
_output_shapes
 "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:џџџџџџџџџ:џџџџџџџџџ : : : 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp:?;
9
_user_specified_name!gru_8/gru_cell/recurrent_kernel:51
/
_user_specified_namegru_8/gru_cell/kernel:3/
-
_user_specified_namegru_8/gru_cell/bias:OK
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_namestates:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ё
m
A__inference_dot_16_layer_call_and_return_conditional_losses_49838
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
:џџџџџџџџџ
Гh
MatMulBatchMatMulV2inputs_0transpose:y:0*
T0*-
_output_shapes
:џџџџџџџџџГГR
ShapeShapeMatMul:output:0*
T0*
_output_shapes
::эЯ]
IdentityIdentityMatMul:output:0*
T0*-
_output_shapes
:џџџџџџџџџГГ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:џџџџџџџџџГ
:џџџџџџџџџГ
:VR
,
_output_shapes
:џџџџџџџџџГ

"
_user_specified_name
inputs_1:V R
,
_output_shapes
:џџџџџџџџџГ

"
_user_specified_name
inputs_0
п
t
H__inference_concatenate_8_layer_call_and_return_conditional_losses_49886
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
:џџџџџџџџџГ\
IdentityIdentityconcat:output:0*
T0*,
_output_shapes
:џџџџџџџџџГ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:џџџџџџџџџГ
:џџџџџџџџџГ
:VR
,
_output_shapes
:џџџџџџџџџГ

"
_user_specified_name
inputs_1:V R
,
_output_shapes
:џџџџџџџџџГ

"
_user_specified_name
inputs_0
і
у
while_cond_49080
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
less_strided_slice_1-
)while_cond_49080___redundant_placeholder0-
)while_cond_49080___redundant_placeholder1-
)while_cond_49080___redundant_placeholder2-
)while_cond_49080___redundant_placeholder3
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
-: : : : :џџџџџџџџџ : :::::
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
:џџџџџџџџџ :
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
ж
o
E__inference_multiply_8_layer_call_and_return_conditional_losses_49363

inputs
inputs_1
identityS
mulMulinputsinputs_1*
T0*,
_output_shapes
:џџџџџџџџџГ
T
IdentityIdentitymul:z:0*
T0*,
_output_shapes
:џџџџџџџџџГ
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:џџџџџџџџџГ
:џџџџџџџџџГ
:TP
,
_output_shapes
:џџџџџџџџџГ

 
_user_specified_nameinputs:T P
,
_output_shapes
:џџџџџџџџџГ

 
_user_specified_nameinputs
ѓ


C__inference_dense_44_layer_call_and_return_conditional_losses_49542

inputs7
%matmul_readvariableop_dense_44_kernel: 2
$biasadd_readvariableop_dense_44_bias:
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOp{
MatMul/ReadVariableOpReadVariableOp%matmul_readvariableop_dense_44_kernel*
_output_shapes

: *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџw
BiasAdd/ReadVariableOpReadVariableOp$biasadd_readvariableop_dense_44_bias*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџV
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџS
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:-)
'
_user_specified_namedense_44/bias:/+
)
_user_specified_namedense_44/kernel:O K
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
Ж
Љ
(__inference_model_52_layer_call_fn_49729
input_prior%
gru_8_gru_cell_bias:`'
gru_8_gru_cell_kernel:`1
gru_8_gru_cell_recurrent_kernel: `!
dense_44_kernel: 
dense_44_bias:
identityЂStatefulPartitionedCallМ
StatefulPartitionedCallStatefulPartitionedCallinput_priorgru_8_gru_cell_biasgru_8_gru_cell_kernelgru_8_gru_cell_recurrent_kerneldense_44_kerneldense_44_bias*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_model_52_layer_call_and_return_conditional_losses_49547o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":џџџџџџџџџГ
: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:-)
'
_user_specified_namedense_44/bias:/+
)
_user_specified_namedense_44/kernel:?;
9
_user_specified_name!gru_8/gru_cell/recurrent_kernel:51
/
_user_specified_namegru_8/gru_cell/kernel:3/
-
_user_specified_namegru_8/gru_cell/bias:Y U
,
_output_shapes
:џџџџџџџџџГ

%
_user_specified_nameinput_prior


(__inference_gru_cell_layer_call_fn_50578

inputs
states_0%
gru_8_gru_cell_bias:`'
gru_8_gru_cell_kernel:`1
gru_8_gru_cell_recurrent_kernel: `
identity

identity_1ЂStatefulPartitionedCallВ
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0gru_8_gru_cell_biasgru_8_gru_cell_kernelgru_8_gru_cell_recurrent_kernel*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:џџџџџџџџџ :џџџџџџџџџ *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_gru_cell_layer_call_and_return_conditional_losses_49208o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:џџџџџџџџџ:џџџџџџџџџ : : : 22
StatefulPartitionedCallStatefulPartitionedCall:?;
9
_user_specified_name!gru_8/gru_cell/recurrent_kernel:51
/
_user_specified_namegru_8/gru_cell/kernel:3/
-
_user_specified_namegru_8/gru_cell/bias:QM
'
_output_shapes
:џџџџџџџџџ 
"
_user_specified_name
states_0:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
й9
К
while_body_49620
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
strided_slice_1_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0?
-gru_cell_readvariableop_gru_8_gru_cell_bias_0:`H
6gru_cell_matmul_readvariableop_gru_8_gru_cell_kernel_0:`T
Bgru_cell_matmul_1_readvariableop_gru_8_gru_cell_recurrent_kernel_0: `
identity

identity_1

identity_2

identity_3

identity_4
strided_slice_1O
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor=
+gru_cell_readvariableop_gru_8_gru_cell_bias:`F
4gru_cell_matmul_readvariableop_gru_8_gru_cell_kernel:`R
@gru_cell_matmul_1_readvariableop_gru_8_gru_cell_recurrent_kernel: `Ђgru_cell/MatMul/ReadVariableOpЂ gru_cell/MatMul_1/ReadVariableOpЂgru_cell/ReadVariableOp
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype0
gru_cell/ReadVariableOpReadVariableOp-gru_cell_readvariableop_gru_8_gru_cell_bias_0*
_output_shapes

:`*
dtype0q
gru_cell/unstackUnpackgru_cell/ReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
num
gru_cell/MatMul/ReadVariableOpReadVariableOp6gru_cell_matmul_readvariableop_gru_8_gru_cell_kernel_0*
_output_shapes

:`*
dtype0
gru_cell/MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0&gru_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ`
gru_cell/BiasAddBiasAddgru_cell/MatMul:product:0gru_cell/unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџ`c
gru_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџО
gru_cell/splitSplit!gru_cell/split/split_dim:output:0gru_cell/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ *
	num_splitЃ
 gru_cell/MatMul_1/ReadVariableOpReadVariableOpBgru_cell_matmul_1_readvariableop_gru_8_gru_cell_recurrent_kernel_0*
_output_shapes

: `*
dtype0
gru_cell/MatMul_1MatMulplaceholder_2(gru_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ`
gru_cell/BiasAdd_1BiasAddgru_cell/MatMul_1:product:0gru_cell/unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџ`c
gru_cell/ConstConst*
_output_shapes
:*
dtype0*!
valueB"        џџџџe
gru_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџъ
gru_cell/split_1SplitVgru_cell/BiasAdd_1:output:0gru_cell/Const:output:0#gru_cell/split_1/split_dim:output:0*

Tlen0*
T0*M
_output_shapes;
9:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ *
	num_split{
gru_cell/addAddV2gru_cell/split:output:0gru_cell/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ _
gru_cell/SigmoidSigmoidgru_cell/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ }
gru_cell/add_1AddV2gru_cell/split:output:1gru_cell/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ c
gru_cell/Sigmoid_1Sigmoidgru_cell/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ x
gru_cell/mulMulgru_cell/Sigmoid_1:y:0gru_cell/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ t
gru_cell/add_2AddV2gru_cell/split:output:2gru_cell/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ [
gru_cell/TanhTanhgru_cell/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ l
gru_cell/mul_1Mulgru_cell/Sigmoid:y:0placeholder_2*
T0*'
_output_shapes
:џџџџџџџџџ S
gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?t
gru_cell/subSubgru_cell/sub/x:output:0gru_cell/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ l
gru_cell/mul_2Mulgru_cell/sub:z:0gru_cell/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ q
gru_cell/add_3AddV2gru_cell/mul_1:z:0gru_cell/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ l
*TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : б
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_13TensorArrayV2Write/TensorListSetItem/index:output:0gru_cell/add_3:z:0*
_output_shapes
: *
element_dtype0:щшвG
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
:џџџџџџџџџ 
NoOpNoOp^gru_cell/MatMul/ReadVariableOp!^gru_cell/MatMul_1/ReadVariableOp^gru_cell/ReadVariableOp*
_output_shapes
 "
@gru_cell_matmul_1_readvariableop_gru_8_gru_cell_recurrent_kernelBgru_cell_matmul_1_readvariableop_gru_8_gru_cell_recurrent_kernel_0"n
4gru_cell_matmul_readvariableop_gru_8_gru_cell_kernel6gru_cell_matmul_readvariableop_gru_8_gru_cell_kernel_0"\
+gru_cell_readvariableop_gru_8_gru_cell_bias-gru_cell_readvariableop_gru_8_gru_cell_bias_0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
identityIdentity:output:0"$
strided_slice_1strided_slice_1_0"
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :џџџџџџџџџ : : : : : 2@
gru_cell/MatMul/ReadVariableOpgru_cell/MatMul/ReadVariableOp2D
 gru_cell/MatMul_1/ReadVariableOp gru_cell/MatMul_1/ReadVariableOp22
gru_cell/ReadVariableOpgru_cell/ReadVariableOp:?	;
9
_user_specified_name!gru_8/gru_cell/recurrent_kernel:51
/
_user_specified_namegru_8/gru_cell/kernel:3/
-
_user_specified_namegru_8/gru_cell/bias:_[
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
:џџџџџџџџџ :
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
і
у
while_cond_49436
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
less_strided_slice_1-
)while_cond_49436___redundant_placeholder0-
)while_cond_49436___redundant_placeholder1-
)while_cond_49436___redundant_placeholder2-
)while_cond_49436___redundant_placeholder3
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
-: : : : :џџџџџџџџџ : :::::
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
:џџџџџџџџџ :
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
я	
у
%__inference_gru_8_layer_call_fn_49902
inputs_0%
gru_8_gru_cell_bias:`'
gru_8_gru_cell_kernel:`1
gru_8_gru_cell_recurrent_kernel: `
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputs_0gru_8_gru_cell_biasgru_8_gru_cell_kernelgru_8_gru_cell_recurrent_kernel*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_gru_8_layer_call_and_return_conditional_losses_49281o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:џџџџџџџџџџџџџџџџџџ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:?;
9
_user_specified_name!gru_8/gru_cell/recurrent_kernel:51
/
_user_specified_namegru_8/gru_cell/kernel:3/
-
_user_specified_namegru_8/gru_cell/bias:^ Z
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs_0"эL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*И
serving_defaultЄ
H
input_prior9
serving_default_input_prior:0џџџџџџџџџГ
<
dense_440
StatefulPartitionedCall:0џџџџџџџџџtensorflow/serving/predict:Їы

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
Ѕ
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
Џ
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
axes"
_tf_keras_layer
Ѕ
 	variables
!trainable_variables
"regularization_losses
#	keras_api
$__call__
*%&call_and_return_all_conditional_losses"
_tf_keras_layer
Џ
&	variables
'trainable_variables
(regularization_losses
)	keras_api
*__call__
*+&call_and_return_all_conditional_losses
,axes"
_tf_keras_layer
Ѕ
-	variables
.trainable_variables
/regularization_losses
0	keras_api
1__call__
*2&call_and_return_all_conditional_losses"
_tf_keras_layer
Ѕ
3	variables
4trainable_variables
5regularization_losses
6	keras_api
7__call__
*8&call_and_return_all_conditional_losses"
_tf_keras_layer
к
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
Л
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
Ъ
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
У
Rtrace_0
Strace_12
(__inference_model_52_layer_call_fn_49729
(__inference_model_52_layer_call_fn_49739Е
ЎВЊ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЂ
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zRtrace_0zStrace_1
љ
Ttrace_0
Utrace_12Т
C__inference_model_52_layer_call_and_return_conditional_losses_49547
C__inference_model_52_layer_call_and_return_conditional_losses_49719Е
ЎВЊ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЂ
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zTtrace_0zUtrace_1
ЯBЬ
 __inference__wrapped_model_49005input_prior"
В
FullArgSpec
args

jargs_0
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
З
Viter

Wbeta_1

Xbeta_2
	Ydecay
Zlearning_rateHmИImЙJmКKmЛLmМHvНIvОJvПKvРLvС"
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
­
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
ч
atrace_02Ъ
-__inference_masking_prior_layer_call_fn_49812
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zatrace_0

btrace_02х
H__inference_masking_prior_layer_call_and_return_conditional_losses_49823
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zbtrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
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
р
htrace_02У
&__inference_dot_16_layer_call_fn_49829
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zhtrace_0
ћ
itrace_02о
A__inference_dot_16_layer_call_and_return_conditional_losses_49838
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zitrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
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
ц
otrace_02Щ
,__inference_activation_8_layer_call_fn_49843
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zotrace_0

ptrace_02ф
G__inference_activation_8_layer_call_and_return_conditional_losses_49848
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zptrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
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
р
vtrace_02У
&__inference_dot_17_layer_call_fn_49854
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zvtrace_0
ћ
wtrace_02о
A__inference_dot_17_layer_call_and_return_conditional_losses_49861
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zwtrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
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
ф
}trace_02Ч
*__inference_multiply_8_layer_call_fn_49867
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 z}trace_0
џ
~trace_02т
E__inference_multiply_8_layer_call_and_return_conditional_losses_49873
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 z~trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Б
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
3	variables
4trainable_variables
5regularization_losses
7__call__
*8&call_and_return_all_conditional_losses
&8"call_and_return_conditional_losses"
_generic_user_object
щ
trace_02Ъ
-__inference_concatenate_8_layer_call_fn_49879
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0

trace_02х
H__inference_concatenate_8_layer_call_and_return_conditional_losses_49886
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0
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
П
states
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
9	variables
:trainable_variables
;regularization_losses
=__call__
*>&call_and_return_all_conditional_losses
&>"call_and_return_conditional_losses"
_generic_user_object
м
trace_0
trace_1
trace_2
trace_32щ
%__inference_gru_8_layer_call_fn_49894
%__inference_gru_8_layer_call_fn_49902
%__inference_gru_8_layer_call_fn_49910
%__inference_gru_8_layer_call_fn_49918Ъ
УВП
FullArgSpec:
args2/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsЂ

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0ztrace_1ztrace_2ztrace_3
Ш
trace_0
trace_1
trace_2
trace_32е
@__inference_gru_8_layer_call_and_return_conditional_losses_50073
@__inference_gru_8_layer_call_and_return_conditional_losses_50228
@__inference_gru_8_layer_call_and_return_conditional_losses_50383
@__inference_gru_8_layer_call_and_return_conditional_losses_50538Ъ
УВП
FullArgSpec:
args2/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsЂ

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0ztrace_1ztrace_2ztrace_3
"
_generic_user_object
я
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
_random_generator

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
В
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
B	variables
Ctrainable_variables
Dregularization_losses
F__call__
*G&call_and_return_all_conditional_losses
&G"call_and_return_conditional_losses"
_generic_user_object
ф
 trace_02Х
(__inference_dense_44_layer_call_fn_50545
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 z trace_0
џ
Ёtrace_02р
C__inference_dense_44_layer_call_and_return_conditional_losses_50556
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zЁtrace_0
!: 2dense_44/kernel
:2dense_44/bias
':%`2gru_8/gru_cell/kernel
1:/ `2gru_8/gru_cell/recurrent_kernel
%:#`2gru_8/gru_cell/bias
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
Ђ0
Ѓ1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ыBш
(__inference_model_52_layer_call_fn_49729input_prior"Ќ
ЅВЁ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ыBш
(__inference_model_52_layer_call_fn_49739input_prior"Ќ
ЅВЁ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
C__inference_model_52_layer_call_and_return_conditional_losses_49547input_prior"Ќ
ЅВЁ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
C__inference_model_52_layer_call_and_return_conditional_losses_49719input_prior"Ќ
ЅВЁ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
:	 (2training_16/Adam/iter
!: (2training_16/Adam/beta_1
!: (2training_16/Adam/beta_2
 : (2training_16/Adam/decay
(:& (2training_16/Adam/learning_rate
зBд
#__inference_signature_wrapper_49807input_prior"
В
FullArgSpec
args 
varargs
 
varkw
 
defaults
  

kwonlyargs
jinput_prior
kwonlydefaults
 
annotationsЊ *
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
зBд
-__inference_masking_prior_layer_call_fn_49812inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ђBя
H__inference_masking_prior_layer_call_and_return_conditional_losses_49823inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
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
мBй
&__inference_dot_16_layer_call_fn_49829inputs_0inputs_1"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
їBє
A__inference_dot_16_layer_call_and_return_conditional_losses_49838inputs_0inputs_1"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
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
жBг
,__inference_activation_8_layer_call_fn_49843inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ёBю
G__inference_activation_8_layer_call_and_return_conditional_losses_49848inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
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
мBй
&__inference_dot_17_layer_call_fn_49854inputs_0inputs_1"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
їBє
A__inference_dot_17_layer_call_and_return_conditional_losses_49861inputs_0inputs_1"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
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
рBн
*__inference_multiply_8_layer_call_fn_49867inputs_0inputs_1"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ћBј
E__inference_multiply_8_layer_call_and_return_conditional_losses_49873inputs_0inputs_1"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
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
уBр
-__inference_concatenate_8_layer_call_fn_49879inputs_0inputs_1"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ўBћ
H__inference_concatenate_8_layer_call_and_return_conditional_losses_49886inputs_0inputs_1"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
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
іBѓ
%__inference_gru_8_layer_call_fn_49894inputs_0"Н
ЖВВ
FullArgSpec:
args2/
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

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
іBѓ
%__inference_gru_8_layer_call_fn_49902inputs_0"Н
ЖВВ
FullArgSpec:
args2/
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

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
єBё
%__inference_gru_8_layer_call_fn_49910inputs"Н
ЖВВ
FullArgSpec:
args2/
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

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
єBё
%__inference_gru_8_layer_call_fn_49918inputs"Н
ЖВВ
FullArgSpec:
args2/
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

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
@__inference_gru_8_layer_call_and_return_conditional_losses_50073inputs_0"Н
ЖВВ
FullArgSpec:
args2/
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

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
@__inference_gru_8_layer_call_and_return_conditional_losses_50228inputs_0"Н
ЖВВ
FullArgSpec:
args2/
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

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
@__inference_gru_8_layer_call_and_return_conditional_losses_50383inputs"Н
ЖВВ
FullArgSpec:
args2/
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

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
@__inference_gru_8_layer_call_and_return_conditional_losses_50538inputs"Н
ЖВВ
FullArgSpec:
args2/
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

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
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
И
Єnon_trainable_variables
Ѕlayers
Іmetrics
 Їlayer_regularization_losses
Јlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
Х
Љtrace_0
Њtrace_12
(__inference_gru_cell_layer_call_fn_50567
(__inference_gru_cell_layer_call_fn_50578Г
ЌВЈ
FullArgSpec+
args# 
jinputs
jstates

jtraining
varargs
 
varkw
 
defaultsЂ
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zЉtrace_0zЊtrace_1
ћ
Ћtrace_0
Ќtrace_12Р
C__inference_gru_cell_layer_call_and_return_conditional_losses_50617
C__inference_gru_cell_layer_call_and_return_conditional_losses_50656Г
ЌВЈ
FullArgSpec+
args# 
jinputs
jstates

jtraining
varargs
 
varkw
 
defaultsЂ
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zЋtrace_0zЌtrace_1
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
вBЯ
(__inference_dense_44_layer_call_fn_50545inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
эBъ
C__inference_dense_44_layer_call_and_return_conditional_losses_50556inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
c
­	variables
Ў	keras_api

Џtotal

Аcount
Б
_fn_kwargs"
_tf_keras_metric

В	variables
Г	keras_api
Дtrue_positives
Еtrue_negatives
Жfalse_positives
Зfalse_negatives"
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
ђBя
(__inference_gru_cell_layer_call_fn_50567inputsstates_0"Ў
ЇВЃ
FullArgSpec+
args# 
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ђBя
(__inference_gru_cell_layer_call_fn_50578inputsstates_0"Ў
ЇВЃ
FullArgSpec+
args# 
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
C__inference_gru_cell_layer_call_and_return_conditional_losses_50617inputsstates_0"Ў
ЇВЃ
FullArgSpec+
args# 
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
C__inference_gru_cell_layer_call_and_return_conditional_losses_50656inputsstates_0"Ў
ЇВЃ
FullArgSpec+
args# 
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
0
Џ0
А1"
trackable_list_wrapper
.
­	variables"
_generic_user_object
:  (2total_8
:  (2count_8
 "
trackable_dict_wrapper
@
Д0
Е1
Ж2
З3"
trackable_list_wrapper
.
В	variables"
_generic_user_object
!:Ш (2true_positives_8
!:Ш (2true_negatives_8
": Ш (2false_positives_8
": Ш (2false_negatives_8
2:0 2"training_16/Adam/dense_44/kernel/m
,:*2 training_16/Adam/dense_44/bias/m
8:6`2(training_16/Adam/gru_8/gru_cell/kernel/m
B:@ `22training_16/Adam/gru_8/gru_cell/recurrent_kernel/m
6:4`2&training_16/Adam/gru_8/gru_cell/bias/m
2:0 2"training_16/Adam/dense_44/kernel/v
,:*2 training_16/Adam/dense_44/bias/v
8:6`2(training_16/Adam/gru_8/gru_cell/kernel/v
B:@ `22training_16/Adam/gru_8/gru_cell/recurrent_kernel/v
6:4`2&training_16/Adam/gru_8/gru_cell/bias/v
 __inference__wrapped_model_49005wLJKHI9Ђ6
/Ђ,
*'
input_priorџџџџџџџџџГ

Њ "3Њ0
.
dense_44"
dense_44џџџџџџџџџЖ
G__inference_activation_8_layer_call_and_return_conditional_losses_49848k5Ђ2
+Ђ(
&#
inputsџџџџџџџџџГГ
Њ "2Ђ/
(%
tensor_0џџџџџџџџџГГ
 
,__inference_activation_8_layer_call_fn_49843`5Ђ2
+Ђ(
&#
inputsџџџџџџџџџГГ
Њ "'$
unknownџџџџџџџџџГГц
H__inference_concatenate_8_layer_call_and_return_conditional_losses_49886dЂa
ZЂW
UR
'$
inputs_0џџџџџџџџџГ

'$
inputs_1џџџџџџџџџГ

Њ "1Ђ.
'$
tensor_0џџџџџџџџџГ
 Р
-__inference_concatenate_8_layer_call_fn_49879dЂa
ZЂW
UR
'$
inputs_0џџџџџџџџџГ

'$
inputs_1џџџџџџџџџГ

Њ "&#
unknownџџџџџџџџџГЊ
C__inference_dense_44_layer_call_and_return_conditional_losses_50556cHI/Ђ,
%Ђ"
 
inputsџџџџџџџџџ 
Њ ",Ђ)
"
tensor_0џџџџџџџџџ
 
(__inference_dense_44_layer_call_fn_50545XHI/Ђ,
%Ђ"
 
inputsџџџџџџџџџ 
Њ "!
unknownџџџџџџџџџр
A__inference_dot_16_layer_call_and_return_conditional_losses_49838dЂa
ZЂW
UR
'$
inputs_0џџџџџџџџџГ

'$
inputs_1џџџџџџџџџГ

Њ "2Ђ/
(%
tensor_0џџџџџџџџџГГ
 К
&__inference_dot_16_layer_call_fn_49829dЂa
ZЂW
UR
'$
inputs_0џџџџџџџџџГ

'$
inputs_1џџџџџџџџџГ

Њ "'$
unknownџџџџџџџџџГГр
A__inference_dot_17_layer_call_and_return_conditional_losses_49861eЂb
[ЂX
VS
(%
inputs_0џџџџџџџџџГГ
'$
inputs_1џџџџџџџџџГ

Њ "1Ђ.
'$
tensor_0џџџџџџџџџГ

 К
&__inference_dot_17_layer_call_fn_49854eЂb
[ЂX
VS
(%
inputs_0џџџџџџџџџГГ
'$
inputs_1џџџџџџџџџГ

Њ "&#
unknownџџџџџџџџџГ
Щ
@__inference_gru_8_layer_call_and_return_conditional_losses_50073LJKOЂL
EЂB
41
/,
inputs_0џџџџџџџџџџџџџџџџџџ

 
p

 
Њ ",Ђ)
"
tensor_0џџџџџџџџџ 
 Щ
@__inference_gru_8_layer_call_and_return_conditional_losses_50228LJKOЂL
EЂB
41
/,
inputs_0џџџџџџџџџџџџџџџџџџ

 
p 

 
Њ ",Ђ)
"
tensor_0џџџџџџџџџ 
 Й
@__inference_gru_8_layer_call_and_return_conditional_losses_50383uLJK@Ђ=
6Ђ3
%"
inputsџџџџџџџџџГ

 
p

 
Њ ",Ђ)
"
tensor_0џџџџџџџџџ 
 Й
@__inference_gru_8_layer_call_and_return_conditional_losses_50538uLJK@Ђ=
6Ђ3
%"
inputsџџџџџџџџџГ

 
p 

 
Њ ",Ђ)
"
tensor_0џџџџџџџџџ 
 Ђ
%__inference_gru_8_layer_call_fn_49894yLJKOЂL
EЂB
41
/,
inputs_0џџџџџџџџџџџџџџџџџџ

 
p

 
Њ "!
unknownџџџџџџџџџ Ђ
%__inference_gru_8_layer_call_fn_49902yLJKOЂL
EЂB
41
/,
inputs_0џџџџџџџџџџџџџџџџџџ

 
p 

 
Њ "!
unknownџџџџџџџџџ 
%__inference_gru_8_layer_call_fn_49910jLJK@Ђ=
6Ђ3
%"
inputsџџџџџџџџџГ

 
p

 
Њ "!
unknownџџџџџџџџџ 
%__inference_gru_8_layer_call_fn_49918jLJK@Ђ=
6Ђ3
%"
inputsџџџџџџџџџГ

 
p 

 
Њ "!
unknownџџџџџџџџџ 
C__inference_gru_cell_layer_call_and_return_conditional_losses_50617ХLJK\ЂY
RЂO
 
inputsџџџџџџџџџ
'Ђ$
"
states_0џџџџџџџџџ 
p
Њ "`Ђ]
VЂS
$!

tensor_0_0џџџџџџџџџ 
+(
&#
tensor_0_1_0џџџџџџџџџ 
 
C__inference_gru_cell_layer_call_and_return_conditional_losses_50656ХLJK\ЂY
RЂO
 
inputsџџџџџџџџџ
'Ђ$
"
states_0џџџџџџџџџ 
p 
Њ "`Ђ]
VЂS
$!

tensor_0_0џџџџџџџџџ 
+(
&#
tensor_0_1_0џџџџџџџџџ 
 ф
(__inference_gru_cell_layer_call_fn_50567ЗLJK\ЂY
RЂO
 
inputsџџџџџџџџџ
'Ђ$
"
states_0џџџџџџџџџ 
p
Њ "RЂO
"
tensor_0џџџџџџџџџ 
)&
$!

tensor_1_0џџџџџџџџџ ф
(__inference_gru_cell_layer_call_fn_50578ЗLJK\ЂY
RЂO
 
inputsџџџџџџџџџ
'Ђ$
"
states_0џџџџџџџџџ 
p 
Њ "RЂO
"
tensor_0џџџџџџџџџ 
)&
$!

tensor_1_0џџџџџџџџџ Е
H__inference_masking_prior_layer_call_and_return_conditional_losses_49823i4Ђ1
*Ђ'
%"
inputsџџџџџџџџџГ

Њ "1Ђ.
'$
tensor_0џџџџџџџџџГ

 
-__inference_masking_prior_layer_call_fn_49812^4Ђ1
*Ђ'
%"
inputsџџџџџџџџџГ

Њ "&#
unknownџџџџџџџџџГ
П
C__inference_model_52_layer_call_and_return_conditional_losses_49547xLJKHIAЂ>
7Ђ4
*'
input_priorџџџџџџџџџГ

p

 
Њ ",Ђ)
"
tensor_0џџџџџџџџџ
 П
C__inference_model_52_layer_call_and_return_conditional_losses_49719xLJKHIAЂ>
7Ђ4
*'
input_priorџџџџџџџџџГ

p 

 
Њ ",Ђ)
"
tensor_0џџџџџџџџџ
 
(__inference_model_52_layer_call_fn_49729mLJKHIAЂ>
7Ђ4
*'
input_priorџџџџџџџџџГ

p

 
Њ "!
unknownџџџџџџџџџ
(__inference_model_52_layer_call_fn_49739mLJKHIAЂ>
7Ђ4
*'
input_priorџџџџџџџџџГ

p 

 
Њ "!
unknownџџџџџџџџџу
E__inference_multiply_8_layer_call_and_return_conditional_losses_49873dЂa
ZЂW
UR
'$
inputs_0џџџџџџџџџГ

'$
inputs_1џџџџџџџџџГ

Њ "1Ђ.
'$
tensor_0џџџџџџџџџГ

 Н
*__inference_multiply_8_layer_call_fn_49867dЂa
ZЂW
UR
'$
inputs_0џџџџџџџџџГ

'$
inputs_1џџџџџџџџџГ

Њ "&#
unknownџџџџџџџџџГ
Ў
#__inference_signature_wrapper_49807LJKHIHЂE
Ђ 
>Њ;
9
input_prior*'
input_priorџџџџџџџџџГ
"3Њ0
.
dense_44"
dense_44џџџџџџџџџ