Г┼
┬"Т"
D
AddV2
x"T
y"T
z"T"
Ttype:
2	АР
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
validate_shapebool( И
p
BatchMatMulV2
x"T
y"T
output"T"
Ttype:
2	"
adj_xbool( "
adj_ybool( 
А
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
resourceИ
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
Ж
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( И
?
Mul
x"T
y"T
z"T"
Ttype:
2	Р

NoOp
U
NotEqual
x"T
y"T
z
"	
Ttype"$
incompatible_shape_errorbool(Р
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
dtypetypeИ
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
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
output"out_typeКэout_type"	
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
М
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
┴
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
executor_typestring Ии
@
StaticRegexFullMatch	
input

output
"
patternstring
ў
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
░
TensorListFromTensor
tensor"element_dtype
element_shape"
shape_type/
output_handleКщшelement_dtype"
element_dtypetype"

shape_typetype:
2	
Я
TensorListReserve
element_shape"
shape_type
num_elements(
handleКщшelement_dtype"
element_dtypetype"

shape_typetype:
2	
И
TensorListStack
input_handle
element_shape
tensor"element_dtype"
element_dtypetype" 
num_elementsint         
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
░
VarHandleOp
resource"
	containerstring "
shared_namestring "

debug_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 И
Ф
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
И
&
	ZerosLike
x"T
y"T"	
Ttype"serve*2.15.02v2.15.0-rc1-8-g6887368d6d48└┤
▐
%training_6/Adam/gru_3/gru_cell/bias/vVarHandleOp*
_output_shapes
: *6

debug_name(&training_6/Adam/gru_3/gru_cell/bias/v/*
dtype0*
shape
:`*6
shared_name'%training_6/Adam/gru_3/gru_cell/bias/v
Я
9training_6/Adam/gru_3/gru_cell/bias/v/Read/ReadVariableOpReadVariableOp%training_6/Adam/gru_3/gru_cell/bias/v*
_output_shapes

:`*
dtype0
В
1training_6/Adam/gru_3/gru_cell/recurrent_kernel/vVarHandleOp*
_output_shapes
: *B

debug_name42training_6/Adam/gru_3/gru_cell/recurrent_kernel/v/*
dtype0*
shape
: `*B
shared_name31training_6/Adam/gru_3/gru_cell/recurrent_kernel/v
╖
Etraining_6/Adam/gru_3/gru_cell/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp1training_6/Adam/gru_3/gru_cell/recurrent_kernel/v*
_output_shapes

: `*
dtype0
ф
'training_6/Adam/gru_3/gru_cell/kernel/vVarHandleOp*
_output_shapes
: *8

debug_name*(training_6/Adam/gru_3/gru_cell/kernel/v/*
dtype0*
shape
:`*8
shared_name)'training_6/Adam/gru_3/gru_cell/kernel/v
г
;training_6/Adam/gru_3/gru_cell/kernel/v/Read/ReadVariableOpReadVariableOp'training_6/Adam/gru_3/gru_cell/kernel/v*
_output_shapes

:`*
dtype0
╚
training_6/Adam/dense_19/bias/vVarHandleOp*
_output_shapes
: *0

debug_name" training_6/Adam/dense_19/bias/v/*
dtype0*
shape:*0
shared_name!training_6/Adam/dense_19/bias/v
П
3training_6/Adam/dense_19/bias/v/Read/ReadVariableOpReadVariableOptraining_6/Adam/dense_19/bias/v*
_output_shapes
:*
dtype0
╥
!training_6/Adam/dense_19/kernel/vVarHandleOp*
_output_shapes
: *2

debug_name$"training_6/Adam/dense_19/kernel/v/*
dtype0*
shape
: *2
shared_name#!training_6/Adam/dense_19/kernel/v
Ч
5training_6/Adam/dense_19/kernel/v/Read/ReadVariableOpReadVariableOp!training_6/Adam/dense_19/kernel/v*
_output_shapes

: *
dtype0
▐
%training_6/Adam/gru_3/gru_cell/bias/mVarHandleOp*
_output_shapes
: *6

debug_name(&training_6/Adam/gru_3/gru_cell/bias/m/*
dtype0*
shape
:`*6
shared_name'%training_6/Adam/gru_3/gru_cell/bias/m
Я
9training_6/Adam/gru_3/gru_cell/bias/m/Read/ReadVariableOpReadVariableOp%training_6/Adam/gru_3/gru_cell/bias/m*
_output_shapes

:`*
dtype0
В
1training_6/Adam/gru_3/gru_cell/recurrent_kernel/mVarHandleOp*
_output_shapes
: *B

debug_name42training_6/Adam/gru_3/gru_cell/recurrent_kernel/m/*
dtype0*
shape
: `*B
shared_name31training_6/Adam/gru_3/gru_cell/recurrent_kernel/m
╖
Etraining_6/Adam/gru_3/gru_cell/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp1training_6/Adam/gru_3/gru_cell/recurrent_kernel/m*
_output_shapes

: `*
dtype0
ф
'training_6/Adam/gru_3/gru_cell/kernel/mVarHandleOp*
_output_shapes
: *8

debug_name*(training_6/Adam/gru_3/gru_cell/kernel/m/*
dtype0*
shape
:`*8
shared_name)'training_6/Adam/gru_3/gru_cell/kernel/m
г
;training_6/Adam/gru_3/gru_cell/kernel/m/Read/ReadVariableOpReadVariableOp'training_6/Adam/gru_3/gru_cell/kernel/m*
_output_shapes

:`*
dtype0
╚
training_6/Adam/dense_19/bias/mVarHandleOp*
_output_shapes
: *0

debug_name" training_6/Adam/dense_19/bias/m/*
dtype0*
shape:*0
shared_name!training_6/Adam/dense_19/bias/m
П
3training_6/Adam/dense_19/bias/m/Read/ReadVariableOpReadVariableOptraining_6/Adam/dense_19/bias/m*
_output_shapes
:*
dtype0
╥
!training_6/Adam/dense_19/kernel/mVarHandleOp*
_output_shapes
: *2

debug_name$"training_6/Adam/dense_19/kernel/m/*
dtype0*
shape
: *2
shared_name#!training_6/Adam/dense_19/kernel/m
Ч
5training_6/Adam/dense_19/kernel/m/Read/ReadVariableOpReadVariableOp!training_6/Adam/dense_19/kernel/m*
_output_shapes

: *
dtype0
Я
false_negatives_3VarHandleOp*
_output_shapes
: *"

debug_namefalse_negatives_3/*
dtype0*
shape:╚*"
shared_namefalse_negatives_3
t
%false_negatives_3/Read/ReadVariableOpReadVariableOpfalse_negatives_3*
_output_shapes	
:╚*
dtype0
Я
false_positives_3VarHandleOp*
_output_shapes
: *"

debug_namefalse_positives_3/*
dtype0*
shape:╚*"
shared_namefalse_positives_3
t
%false_positives_3/Read/ReadVariableOpReadVariableOpfalse_positives_3*
_output_shapes	
:╚*
dtype0
Ь
true_negatives_3VarHandleOp*
_output_shapes
: *!

debug_nametrue_negatives_3/*
dtype0*
shape:╚*!
shared_nametrue_negatives_3
r
$true_negatives_3/Read/ReadVariableOpReadVariableOptrue_negatives_3*
_output_shapes	
:╚*
dtype0
Ь
true_positives_3VarHandleOp*
_output_shapes
: *!

debug_nametrue_positives_3/*
dtype0*
shape:╚*!
shared_nametrue_positives_3
r
$true_positives_3/Read/ReadVariableOpReadVariableOptrue_positives_3*
_output_shapes	
:╚*
dtype0
|
count_3VarHandleOp*
_output_shapes
: *

debug_name
count_3/*
dtype0*
shape: *
shared_name	count_3
[
count_3/Read/ReadVariableOpReadVariableOpcount_3*
_output_shapes
: *
dtype0
|
total_3VarHandleOp*
_output_shapes
: *

debug_name
total_3/*
dtype0*
shape: *
shared_name	total_3
[
total_3/Read/ReadVariableOpReadVariableOptotal_3*
_output_shapes
: *
dtype0
╛
training_6/Adam/learning_rateVarHandleOp*
_output_shapes
: *.

debug_name training_6/Adam/learning_rate/*
dtype0*
shape: *.
shared_nametraining_6/Adam/learning_rate
З
1training_6/Adam/learning_rate/Read/ReadVariableOpReadVariableOptraining_6/Adam/learning_rate*
_output_shapes
: *
dtype0
ж
training_6/Adam/decayVarHandleOp*
_output_shapes
: *&

debug_nametraining_6/Adam/decay/*
dtype0*
shape: *&
shared_nametraining_6/Adam/decay
w
)training_6/Adam/decay/Read/ReadVariableOpReadVariableOptraining_6/Adam/decay*
_output_shapes
: *
dtype0
й
training_6/Adam/beta_2VarHandleOp*
_output_shapes
: *'

debug_nametraining_6/Adam/beta_2/*
dtype0*
shape: *'
shared_nametraining_6/Adam/beta_2
y
*training_6/Adam/beta_2/Read/ReadVariableOpReadVariableOptraining_6/Adam/beta_2*
_output_shapes
: *
dtype0
й
training_6/Adam/beta_1VarHandleOp*
_output_shapes
: *'

debug_nametraining_6/Adam/beta_1/*
dtype0*
shape: *'
shared_nametraining_6/Adam/beta_1
y
*training_6/Adam/beta_1/Read/ReadVariableOpReadVariableOptraining_6/Adam/beta_1*
_output_shapes
: *
dtype0
г
training_6/Adam/iterVarHandleOp*
_output_shapes
: *%

debug_nametraining_6/Adam/iter/*
dtype0	*
shape: *%
shared_nametraining_6/Adam/iter
u
(training_6/Adam/iter/Read/ReadVariableOpReadVariableOptraining_6/Adam/iter*
_output_shapes
: *
dtype0	
и
gru_3/gru_cell/biasVarHandleOp*
_output_shapes
: *$

debug_namegru_3/gru_cell/bias/*
dtype0*
shape
:`*$
shared_namegru_3/gru_cell/bias
{
'gru_3/gru_cell/bias/Read/ReadVariableOpReadVariableOpgru_3/gru_cell/bias*
_output_shapes

:`*
dtype0
╠
gru_3/gru_cell/recurrent_kernelVarHandleOp*
_output_shapes
: *0

debug_name" gru_3/gru_cell/recurrent_kernel/*
dtype0*
shape
: `*0
shared_name!gru_3/gru_cell/recurrent_kernel
У
3gru_3/gru_cell/recurrent_kernel/Read/ReadVariableOpReadVariableOpgru_3/gru_cell/recurrent_kernel*
_output_shapes

: `*
dtype0
о
gru_3/gru_cell/kernelVarHandleOp*
_output_shapes
: *&

debug_namegru_3/gru_cell/kernel/*
dtype0*
shape
:`*&
shared_namegru_3/gru_cell/kernel

)gru_3/gru_cell/kernel/Read/ReadVariableOpReadVariableOpgru_3/gru_cell/kernel*
_output_shapes

:`*
dtype0
Т
dense_19/biasVarHandleOp*
_output_shapes
: *

debug_namedense_19/bias/*
dtype0*
shape:*
shared_namedense_19/bias
k
!dense_19/bias/Read/ReadVariableOpReadVariableOpdense_19/bias*
_output_shapes
:*
dtype0
Ь
dense_19/kernelVarHandleOp*
_output_shapes
: * 

debug_namedense_19/kernel/*
dtype0*
shape
: * 
shared_namedense_19/kernel
s
#dense_19/kernel/Read/ReadVariableOpReadVariableOpdense_19/kernel*
_output_shapes

: *
dtype0
И
serving_default_input_priorPlaceholder*,
_output_shapes
:         │
*
dtype0*!
shape:         │

м
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_priorgru_3/gru_cell/biasgru_3/gru_cell/kernelgru_3/gru_cell/recurrent_kerneldense_19/kerneldense_19/bias*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8В *,
f'R%
#__inference_signature_wrapper_21467

NoOpNoOp
єI
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*оI
valueдIBбI BЪI
ї
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
О
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses* 
Ш
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
axes* 
О
 	variables
!trainable_variables
"regularization_losses
#	keras_api
$__call__
*%&call_and_return_all_conditional_losses* 
Ш
&	variables
'trainable_variables
(regularization_losses
)	keras_api
*__call__
*+&call_and_return_all_conditional_losses
,axes* 
О
-	variables
.trainable_variables
/regularization_losses
0	keras_api
1__call__
*2&call_and_return_all_conditional_losses* 
О
3	variables
4trainable_variables
5regularization_losses
6	keras_api
7__call__
*8&call_and_return_all_conditional_losses* 
┴
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
ж
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
░
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
и
Viter

Wbeta_1

Xbeta_2
	Ydecay
Zlearning_rateHm╕Im╣Jm║Km╗Lm╝Hv╜Iv╛Jv┐Kv└Lv┴*

[serving_default* 
* 
* 
* 
С
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
С
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
С
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
С
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
С
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
Х
non_trainable_variables
Аlayers
Бmetrics
 Вlayer_regularization_losses
Гlayer_metrics
3	variables
4trainable_variables
5regularization_losses
7__call__
*8&call_and_return_all_conditional_losses
&8"call_and_return_conditional_losses* 

Дtrace_0* 

Еtrace_0* 

J0
K1
L2*

J0
K1
L2*
* 
е
Жstates
Зnon_trainable_variables
Иlayers
Йmetrics
 Кlayer_regularization_losses
Лlayer_metrics
9	variables
:trainable_variables
;regularization_losses
=__call__
*>&call_and_return_all_conditional_losses
&>"call_and_return_conditional_losses*
:
Мtrace_0
Нtrace_1
Оtrace_2
Пtrace_3* 
:
Рtrace_0
Сtrace_1
Тtrace_2
Уtrace_3* 
* 
┌
Ф	variables
Хtrainable_variables
Цregularization_losses
Ч	keras_api
Ш__call__
+Щ&call_and_return_all_conditional_losses
Ъ_random_generator

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
Ш
Ыnon_trainable_variables
Ьlayers
Эmetrics
 Юlayer_regularization_losses
Яlayer_metrics
B	variables
Ctrainable_variables
Dregularization_losses
F__call__
*G&call_and_return_all_conditional_losses
&G"call_and_return_conditional_losses*

аtrace_0* 

бtrace_0* 
_Y
VARIABLE_VALUEdense_19/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_19/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEgru_3/gru_cell/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEgru_3/gru_cell/recurrent_kernel&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEgru_3/gru_cell/bias&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
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
в0
г1*
* 
* 
* 
* 
* 
* 
WQ
VARIABLE_VALUEtraining_6/Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEtraining_6/Adam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEtraining_6/Adam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEtraining_6/Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEtraining_6/Adam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
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
Ю
дnon_trainable_variables
еlayers
жmetrics
 зlayer_regularization_losses
иlayer_metrics
Ф	variables
Хtrainable_variables
Цregularization_losses
Ш__call__
+Щ&call_and_return_all_conditional_losses
'Щ"call_and_return_conditional_losses*

йtrace_0
кtrace_1* 

лtrace_0
мtrace_1* 
* 
* 
* 
* 
* 
* 
* 
* 
M
н	variables
о	keras_api

пtotal

░count
▒
_fn_kwargs*
z
▓	variables
│	keras_api
┤true_positives
╡true_negatives
╢false_positives
╖false_negatives*
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
п0
░1*

н	variables*
UO
VARIABLE_VALUEtotal_34keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_34keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*
* 
$
┤0
╡1
╢2
╖3*

▓	variables*
ga
VARIABLE_VALUEtrue_positives_3=keras_api/metrics/1/true_positives/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUEtrue_negatives_3=keras_api/metrics/1/true_negatives/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEfalse_positives_3>keras_api/metrics/1/false_positives/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEfalse_negatives_3>keras_api/metrics/1/false_negatives/.ATTRIBUTES/VARIABLE_VALUE*
ОЗ
VARIABLE_VALUE!training_6/Adam/dense_19/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
КГ
VARIABLE_VALUEtraining_6/Adam/dense_19/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
Г}
VARIABLE_VALUE'training_6/Adam/gru_3/gru_cell/kernel/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
ОЗ
VARIABLE_VALUE1training_6/Adam/gru_3/gru_cell/recurrent_kernel/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
Б{
VARIABLE_VALUE%training_6/Adam/gru_3/gru_cell/bias/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
ОЗ
VARIABLE_VALUE!training_6/Adam/dense_19/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
КГ
VARIABLE_VALUEtraining_6/Adam/dense_19/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
Г}
VARIABLE_VALUE'training_6/Adam/gru_3/gru_cell/kernel/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
ОЗ
VARIABLE_VALUE1training_6/Adam/gru_3/gru_cell/recurrent_kernel/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
Б{
VARIABLE_VALUE%training_6/Adam/gru_3/gru_cell/bias/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
З
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenamedense_19/kerneldense_19/biasgru_3/gru_cell/kernelgru_3/gru_cell/recurrent_kernelgru_3/gru_cell/biastraining_6/Adam/itertraining_6/Adam/beta_1training_6/Adam/beta_2training_6/Adam/decaytraining_6/Adam/learning_ratetotal_3count_3true_positives_3true_negatives_3false_positives_3false_negatives_3!training_6/Adam/dense_19/kernel/mtraining_6/Adam/dense_19/bias/m'training_6/Adam/gru_3/gru_cell/kernel/m1training_6/Adam/gru_3/gru_cell/recurrent_kernel/m%training_6/Adam/gru_3/gru_cell/bias/m!training_6/Adam/dense_19/kernel/vtraining_6/Adam/dense_19/bias/v'training_6/Adam/gru_3/gru_cell/kernel/v1training_6/Adam/gru_3/gru_cell/recurrent_kernel/v%training_6/Adam/gru_3/gru_cell/bias/vConst*'
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
GPU 2J 8В *'
f"R 
__inference__traced_save_22494
В
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_19/kerneldense_19/biasgru_3/gru_cell/kernelgru_3/gru_cell/recurrent_kernelgru_3/gru_cell/biastraining_6/Adam/itertraining_6/Adam/beta_1training_6/Adam/beta_2training_6/Adam/decaytraining_6/Adam/learning_ratetotal_3count_3true_positives_3true_negatives_3false_positives_3false_negatives_3!training_6/Adam/dense_19/kernel/mtraining_6/Adam/dense_19/bias/m'training_6/Adam/gru_3/gru_cell/kernel/m1training_6/Adam/gru_3/gru_cell/recurrent_kernel/m%training_6/Adam/gru_3/gru_cell/bias/m!training_6/Adam/dense_19/kernel/vtraining_6/Adam/dense_19/bias/v'training_6/Adam/gru_3/gru_cell/kernel/v1training_6/Adam/gru_3/gru_cell/recurrent_kernel/v%training_6/Adam/gru_3/gru_cell/bias/v*&
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
GPU 2J 8В **
f%R#
!__inference__traced_restore_22581иУ
┘	
с
%__inference_gru_3_layer_call_fn_21570

inputs%
gru_3_gru_cell_bias:`'
gru_3_gru_cell_kernel:`1
gru_3_gru_cell_recurrent_kernel: `
identityИвStatefulPartitionedCallР
StatefulPartitionedCallStatefulPartitionedCallinputsgru_3_gru_cell_biasgru_3_gru_cell_kernelgru_3_gru_cell_recurrent_kernel*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:          *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *I
fDRB
@__inference_gru_3_layer_call_and_return_conditional_losses_21187o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:          <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:         │: : : 22
StatefulPartitionedCallStatefulPartitionedCall:?;
9
_user_specified_name!gru_3/gru_cell/recurrent_kernel:51
/
_user_specified_namegru_3/gru_cell/kernel:3/
-
_user_specified_namegru_3/gru_cell/bias:T P
,
_output_shapes
:         │
 
_user_specified_nameinputs
┘9
║
while_body_21643
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
strided_slice_1_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0?
-gru_cell_readvariableop_gru_3_gru_cell_bias_0:`H
6gru_cell_matmul_readvariableop_gru_3_gru_cell_kernel_0:`T
Bgru_cell_matmul_1_readvariableop_gru_3_gru_cell_recurrent_kernel_0: `
identity

identity_1

identity_2

identity_3

identity_4
strided_slice_1O
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor=
+gru_cell_readvariableop_gru_3_gru_cell_bias:`F
4gru_cell_matmul_readvariableop_gru_3_gru_cell_kernel:`R
@gru_cell_matmul_1_readvariableop_gru_3_gru_cell_recurrent_kernel: `Ивgru_cell/MatMul/ReadVariableOpв gru_cell/MatMul_1/ReadVariableOpвgru_cell/ReadVariableOpВ
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       О
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0Е
gru_cell/ReadVariableOpReadVariableOp-gru_cell_readvariableop_gru_3_gru_cell_bias_0*
_output_shapes

:`*
dtype0q
gru_cell/unstackUnpackgru_cell/ReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
numХ
gru_cell/MatMul/ReadVariableOpReadVariableOp6gru_cell_matmul_readvariableop_gru_3_gru_cell_kernel_0*
_output_shapes

:`*
dtype0Я
gru_cell/MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0&gru_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         `Г
gru_cell/BiasAddBiasAddgru_cell/MatMul:product:0gru_cell/unstack:output:0*
T0*'
_output_shapes
:         `c
gru_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         ╛
gru_cell/splitSplit!gru_cell/split/split_dim:output:0gru_cell/BiasAdd:output:0*
T0*M
_output_shapes;
9:          :          :          *
	num_splitг
 gru_cell/MatMul_1/ReadVariableOpReadVariableOpBgru_cell_matmul_1_readvariableop_gru_3_gru_cell_recurrent_kernel_0*
_output_shapes

: `*
dtype0Ж
gru_cell/MatMul_1MatMulplaceholder_2(gru_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         `З
gru_cell/BiasAdd_1BiasAddgru_cell/MatMul_1:product:0gru_cell/unstack:output:1*
T0*'
_output_shapes
:         `c
gru_cell/ConstConst*
_output_shapes
:*
dtype0*!
valueB"            e
gru_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         ъ
gru_cell/split_1SplitVgru_cell/BiasAdd_1:output:0gru_cell/Const:output:0#gru_cell/split_1/split_dim:output:0*

Tlen0*
T0*M
_output_shapes;
9:          :          :          *
	num_split{
gru_cell/addAddV2gru_cell/split:output:0gru_cell/split_1:output:0*
T0*'
_output_shapes
:          _
gru_cell/SigmoidSigmoidgru_cell/add:z:0*
T0*'
_output_shapes
:          }
gru_cell/add_1AddV2gru_cell/split:output:1gru_cell/split_1:output:1*
T0*'
_output_shapes
:          c
gru_cell/Sigmoid_1Sigmoidgru_cell/add_1:z:0*
T0*'
_output_shapes
:          x
gru_cell/mulMulgru_cell/Sigmoid_1:y:0gru_cell/split_1:output:2*
T0*'
_output_shapes
:          t
gru_cell/add_2AddV2gru_cell/split:output:2gru_cell/mul:z:0*
T0*'
_output_shapes
:          [
gru_cell/TanhTanhgru_cell/add_2:z:0*
T0*'
_output_shapes
:          l
gru_cell/mul_1Mulgru_cell/Sigmoid:y:0placeholder_2*
T0*'
_output_shapes
:          S
gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?t
gru_cell/subSubgru_cell/sub/x:output:0gru_cell/Sigmoid:y:0*
T0*'
_output_shapes
:          l
gru_cell/mul_2Mulgru_cell/sub:z:0gru_cell/Tanh:y:0*
T0*'
_output_shapes
:          q
gru_cell/add_3AddV2gru_cell/mul_1:z:0gru_cell/mul_2:z:0*
T0*'
_output_shapes
:          l
*TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : ╤
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_13TensorArrayV2Write/TensorListSetItem/index:output:0gru_cell/add_3:z:0*
_output_shapes
: *
element_dtype0:щш╥G
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
:          А
NoOpNoOp^gru_cell/MatMul/ReadVariableOp!^gru_cell/MatMul_1/ReadVariableOp^gru_cell/ReadVariableOp*
_output_shapes
 "Ж
@gru_cell_matmul_1_readvariableop_gru_3_gru_cell_recurrent_kernelBgru_cell_matmul_1_readvariableop_gru_3_gru_cell_recurrent_kernel_0"n
4gru_cell_matmul_readvariableop_gru_3_gru_cell_kernel6gru_cell_matmul_readvariableop_gru_3_gru_cell_kernel_0"\
+gru_cell_readvariableop_gru_3_gru_cell_bias-gru_cell_readvariableop_gru_3_gru_cell_bias_0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
identityIdentity:output:0"$
strided_slice_1strided_slice_1_0"Ь
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :          : : : : : 2@
gru_cell/MatMul/ReadVariableOpgru_cell/MatMul/ReadVariableOp2D
 gru_cell/MatMul_1/ReadVariableOp gru_cell/MatMul_1/ReadVariableOp22
gru_cell/ReadVariableOpgru_cell/ReadVariableOp:?	;
9
_user_specified_name!gru_3/gru_cell/recurrent_kernel:51
/
_user_specified_namegru_3/gru_cell/kernel:3/
-
_user_specified_namegru_3/gru_cell/bias:_[
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
:          :
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

м
model_22_gru_3_while_cond_20552%
!model_22_gru_3_while_loop_counter+
'model_22_gru_3_while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3'
#less_model_22_gru_3_strided_slice_1<
8model_22_gru_3_while_cond_20552___redundant_placeholder0<
8model_22_gru_3_while_cond_20552___redundant_placeholder1<
8model_22_gru_3_while_cond_20552___redundant_placeholder2<
8model_22_gru_3_while_cond_20552___redundant_placeholder3<
8model_22_gru_3_while_cond_20552___redundant_placeholder4
identity
_
LessLessplaceholder#less_model_22_gru_3_strided_slice_1*
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
D: : : : :          :          : ::::::
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
_user_specified_name model_22/gru_3/strided_slice_1:-)
'
_output_shapes
:          :-)
'
_output_shapes
:          :
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
_user_specified_name)'model_22/gru_3/while/maximum_iterations:Y U

_output_shapes
: 
;
_user_specified_name#!model_22/gru_3/while/loop_counter
╩
d
H__inference_masking_prior_layer_call_and_return_conditional_losses_20992

inputs
identityO

NotEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А┐h
NotEqualNotEqualinputsNotEqual/y:output:0*
T0*,
_output_shapes
:         │
`
Any/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
         w
AnyAnyNotEqual:z:0Any/reduction_indices:output:0*,
_output_shapes
:         │*
	keep_dims(`
CastCastAny:output:0*

DstT0*

SrcT0
*,
_output_shapes
:         │S
mulMulinputsCast:y:0*
T0*,
_output_shapes
:         │
s
SqueezeSqueezeAny:output:0*
T0
*(
_output_shapes
:         │*
squeeze_dims

         T
IdentityIdentitymul:z:0*
T0*,
_output_shapes
:         │
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         │
:T P
,
_output_shapes
:         │

 
_user_specified_nameinputs
▒!
Е
while_body_20879
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
strided_slice_1_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_00
gru_cell_gru_3_gru_cell_bias_0:`2
 gru_cell_gru_3_gru_cell_kernel_0:`<
*gru_cell_gru_3_gru_cell_recurrent_kernel_0: `
identity

identity_1

identity_2

identity_3

identity_4
strided_slice_1O
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor.
gru_cell_gru_3_gru_cell_bias:`0
gru_cell_gru_3_gru_cell_kernel:`:
(gru_cell_gru_3_gru_cell_recurrent_kernel: `Ив gru_cell/StatefulPartitionedCallВ
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       О
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0Е
 gru_cell/StatefulPartitionedCallStatefulPartitionedCall*TensorArrayV2Read/TensorListGetItem:item:0placeholder_2gru_cell_gru_3_gru_cell_bias_0 gru_cell_gru_3_gru_cell_kernel_0*gru_cell_gru_3_gru_cell_recurrent_kernel_0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:          :          *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_gru_cell_layer_call_and_return_conditional_losses_20868l
*TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : ш
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_13TensorArrayV2Write/TensorListSetItem/index:output:0)gru_cell/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:щш╥G
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
:          E
NoOpNoOp!^gru_cell/StatefulPartitionedCall*
_output_shapes
 ">
gru_cell_gru_3_gru_cell_biasgru_cell_gru_3_gru_cell_bias_0"B
gru_cell_gru_3_gru_cell_kernel gru_cell_gru_3_gru_cell_kernel_0"V
(gru_cell_gru_3_gru_cell_recurrent_kernel*gru_cell_gru_3_gru_cell_recurrent_kernel_0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
identityIdentity:output:0"$
strided_slice_1strided_slice_1_0"Ь
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :          : : : : : 2D
 gru_cell/StatefulPartitionedCall gru_cell/StatefulPartitionedCall:?	;
9
_user_specified_name!gru_3/gru_cell/recurrent_kernel:51
/
_user_specified_namegru_3/gru_cell/kernel:3/
-
_user_specified_namegru_3/gru_cell/bias:_[
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
:          :
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
Г
C__inference_gru_cell_layer_call_and_return_conditional_losses_22277

inputs
states_04
"readvariableop_gru_3_gru_cell_bias:`=
+matmul_readvariableop_gru_3_gru_cell_kernel:`I
7matmul_1_readvariableop_gru_3_gru_cell_recurrent_kernel: `
identity

identity_1ИвMatMul/ReadVariableOpвMatMul_1/ReadVariableOpвReadVariableOpq
ReadVariableOpReadVariableOp"readvariableop_gru_3_gru_cell_bias*
_output_shapes

:`*
dtype0_
unstackUnpackReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
numБ
MatMul/ReadVariableOpReadVariableOp+matmul_readvariableop_gru_3_gru_cell_kernel*
_output_shapes

:`*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         `h
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*'
_output_shapes
:         `Z
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         г
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*M
_output_shapes;
9:          :          :          *
	num_splitП
MatMul_1/ReadVariableOpReadVariableOp7matmul_1_readvariableop_gru_3_gru_cell_recurrent_kernel*
_output_shapes

: `*
dtype0o
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         `l
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*'
_output_shapes
:         `Z
ConstConst*
_output_shapes
:*
dtype0*!
valueB"            \
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         ╞
split_1SplitVBiasAdd_1:output:0Const:output:0split_1/split_dim:output:0*

Tlen0*
T0*M
_output_shapes;
9:          :          :          *
	num_split`
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:          M
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:          b
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:          Q
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:          ]
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:          Y
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:          I
TanhTanh	add_2:z:0*
T0*'
_output_shapes
:          U
mul_1MulSigmoid:y:0states_0*
T0*'
_output_shapes
:          J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:          Q
mul_2Mulsub:z:0Tanh:y:0*
T0*'
_output_shapes
:          V
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*'
_output_shapes
:          X
IdentityIdentity	add_3:z:0^NoOp*
T0*'
_output_shapes
:          Z

Identity_1Identity	add_3:z:0^NoOp*
T0*'
_output_shapes
:          e
NoOpNoOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
_output_shapes
 "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:         :          : : : 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp:?;
9
_user_specified_name!gru_3/gru_cell/recurrent_kernel:51
/
_user_specified_namegru_3/gru_cell/kernel:3/
-
_user_specified_namegru_3/gru_cell/bias:QM
'
_output_shapes
:          
"
_user_specified_name
states_0:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
РЭ
╘
 __inference__wrapped_model_20665
input_priorL
:model_22_gru_3_gru_cell_readvariableop_gru_3_gru_cell_bias:`U
Cmodel_22_gru_3_gru_cell_matmul_readvariableop_gru_3_gru_cell_kernel:`a
Omodel_22_gru_3_gru_cell_matmul_1_readvariableop_gru_3_gru_cell_recurrent_kernel: `I
7model_22_dense_19_matmul_readvariableop_dense_19_kernel: D
6model_22_dense_19_biasadd_readvariableop_dense_19_bias:
identityИв(model_22/dense_19/BiasAdd/ReadVariableOpв'model_22/dense_19/MatMul/ReadVariableOpв-model_22/gru_3/gru_cell/MatMul/ReadVariableOpв/model_22/gru_3/gru_cell/MatMul_1/ReadVariableOpв&model_22/gru_3/gru_cell/ReadVariableOpвmodel_22/gru_3/whilef
!model_22/masking_prior/NotEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А┐Ы
model_22/masking_prior/NotEqualNotEqualinput_prior*model_22/masking_prior/NotEqual/y:output:0*
T0*,
_output_shapes
:         │
w
,model_22/masking_prior/Any/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
         ╝
model_22/masking_prior/AnyAny#model_22/masking_prior/NotEqual:z:05model_22/masking_prior/Any/reduction_indices:output:0*,
_output_shapes
:         │*
	keep_dims(О
model_22/masking_prior/CastCast#model_22/masking_prior/Any:output:0*

DstT0*

SrcT0
*,
_output_shapes
:         │Ж
model_22/masking_prior/mulMulinput_priormodel_22/masking_prior/Cast:y:0*
T0*,
_output_shapes
:         │
б
model_22/masking_prior/SqueezeSqueeze#model_22/masking_prior/Any:output:0*
T0
*(
_output_shapes
:         │*
squeeze_dims

         r
model_22/dot_6/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          д
model_22/dot_6/transpose	Transposemodel_22/masking_prior/mul:z:0&model_22/dot_6/transpose/perm:output:0*
T0*,
_output_shapes
:         
│Ь
model_22/dot_6/MatMulBatchMatMulV2model_22/masking_prior/mul:z:0model_22/dot_6/transpose:y:0*
T0*-
_output_shapes
:         ││p
model_22/dot_6/ShapeShapemodel_22/dot_6/MatMul:output:0*
T0*
_output_shapes
::э╧А
model_22/activation_3/SoftmaxSoftmaxmodel_22/dot_6/MatMul:output:0*
T0*-
_output_shapes
:         ││ж
model_22/dot_7/MatMulBatchMatMulV2'model_22/activation_3/Softmax:softmax:0model_22/masking_prior/mul:z:0*
T0*,
_output_shapes
:         │
p
model_22/dot_7/ShapeShapemodel_22/dot_7/MatMul:output:0*
T0*
_output_shapes
::э╧Х
model_22/multiply_3/mulMulmodel_22/dot_7/MatMul:output:0model_22/masking_prior/mul:z:0*
T0*,
_output_shapes
:         │
d
"model_22/multiply_3/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : ╣
model_22/multiply_3/ExpandDims
ExpandDims'model_22/masking_prior/Squeeze:output:0+model_22/multiply_3/ExpandDims/dim:output:0*
T0
*,
_output_shapes
:         │g
%model_22/multiply_3/concat/concat_dimConst*
_output_shapes
: *
dtype0*
value	B : Н
!model_22/multiply_3/concat/concatIdentity'model_22/multiply_3/ExpandDims:output:0*
T0
*,
_output_shapes
:         │k
)model_22/multiply_3/All/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : и
model_22/multiply_3/AllAll*model_22/multiply_3/concat/concat:output:02model_22/multiply_3/All/reduction_indices:output:0*(
_output_shapes
:         │d
"model_22/concatenate_3/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :╙
model_22/concatenate_3/concatConcatV2model_22/masking_prior/mul:z:0model_22/multiply_3/mul:z:0+model_22/concatenate_3/concat/axis:output:0*
N*
T0*,
_output_shapes
:         │p
%model_22/concatenate_3/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
         ┐
!model_22/concatenate_3/ExpandDims
ExpandDims'model_22/masking_prior/Squeeze:output:0.model_22/concatenate_3/ExpandDims/dim:output:0*
T0
*,
_output_shapes
:         │r
'model_22/concatenate_3/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
valueB :
         ╝
#model_22/concatenate_3/ExpandDims_1
ExpandDims model_22/multiply_3/All:output:00model_22/concatenate_3/ExpandDims_1/dim:output:0*
T0
*,
_output_shapes
:         │f
$model_22/concatenate_3/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B :Ї
model_22/concatenate_3/concat_1ConcatV2*model_22/concatenate_3/ExpandDims:output:0,model_22/concatenate_3/ExpandDims_1:output:0-model_22/concatenate_3/concat_1/axis:output:0*
N*
T0
*,
_output_shapes
:         │w
,model_22/concatenate_3/All/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
         м
model_22/concatenate_3/AllAll(model_22/concatenate_3/concat_1:output:05model_22/concatenate_3/All/reduction_indices:output:0*(
_output_shapes
:         │x
model_22/gru_3/ShapeShape&model_22/concatenate_3/concat:output:0*
T0*
_output_shapes
::э╧l
"model_22/gru_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: n
$model_22/gru_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:n
$model_22/gru_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ь
model_22/gru_3/strided_sliceStridedSlicemodel_22/gru_3/Shape:output:0+model_22/gru_3/strided_slice/stack:output:0-model_22/gru_3/strided_slice/stack_1:output:0-model_22/gru_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
model_22/gru_3/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : а
model_22/gru_3/zeros/packedPack%model_22/gru_3/strided_slice:output:0&model_22/gru_3/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:_
model_22/gru_3/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    Щ
model_22/gru_3/zerosFill$model_22/gru_3/zeros/packed:output:0#model_22/gru_3/zeros/Const:output:0*
T0*'
_output_shapes
:          r
model_22/gru_3/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          м
model_22/gru_3/transpose	Transpose&model_22/concatenate_3/concat:output:0&model_22/gru_3/transpose/perm:output:0*
T0*,
_output_shapes
:│         p
model_22/gru_3/Shape_1Shapemodel_22/gru_3/transpose:y:0*
T0*
_output_shapes
::э╧n
$model_22/gru_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: p
&model_22/gru_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:p
&model_22/gru_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ж
model_22/gru_3/strided_slice_1StridedSlicemodel_22/gru_3/Shape_1:output:0-model_22/gru_3/strided_slice_1/stack:output:0/model_22/gru_3/strided_slice_1/stack_1:output:0/model_22/gru_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskh
model_22/gru_3/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
         л
model_22/gru_3/ExpandDims
ExpandDims#model_22/concatenate_3/All:output:0&model_22/gru_3/ExpandDims/dim:output:0*
T0
*,
_output_shapes
:         │t
model_22/gru_3/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          м
model_22/gru_3/transpose_1	Transpose"model_22/gru_3/ExpandDims:output:0(model_22/gru_3/transpose_1/perm:output:0*
T0
*,
_output_shapes
:│         u
*model_22/gru_3/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         с
model_22/gru_3/TensorArrayV2TensorListReserve3model_22/gru_3/TensorArrayV2/element_shape:output:0'model_22/gru_3/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥Х
Dmodel_22/gru_3/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       Н
6model_22/gru_3/TensorArrayUnstack/TensorListFromTensorTensorListFromTensormodel_22/gru_3/transpose:y:0Mmodel_22/gru_3/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥n
$model_22/gru_3/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: p
&model_22/gru_3/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:p
&model_22/gru_3/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:┤
model_22/gru_3/strided_slice_2StridedSlicemodel_22/gru_3/transpose:y:0-model_22/gru_3/strided_slice_2/stack:output:0/model_22/gru_3/strided_slice_2/stack_1:output:0/model_22/gru_3/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_maskб
&model_22/gru_3/gru_cell/ReadVariableOpReadVariableOp:model_22_gru_3_gru_cell_readvariableop_gru_3_gru_cell_bias*
_output_shapes

:`*
dtype0П
model_22/gru_3/gru_cell/unstackUnpack.model_22/gru_3/gru_cell/ReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
num▒
-model_22/gru_3/gru_cell/MatMul/ReadVariableOpReadVariableOpCmodel_22_gru_3_gru_cell_matmul_readvariableop_gru_3_gru_cell_kernel*
_output_shapes

:`*
dtype0║
model_22/gru_3/gru_cell/MatMulMatMul'model_22/gru_3/strided_slice_2:output:05model_22/gru_3/gru_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         `░
model_22/gru_3/gru_cell/BiasAddBiasAdd(model_22/gru_3/gru_cell/MatMul:product:0(model_22/gru_3/gru_cell/unstack:output:0*
T0*'
_output_shapes
:         `r
'model_22/gru_3/gru_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         ы
model_22/gru_3/gru_cell/splitSplit0model_22/gru_3/gru_cell/split/split_dim:output:0(model_22/gru_3/gru_cell/BiasAdd:output:0*
T0*M
_output_shapes;
9:          :          :          *
	num_split┐
/model_22/gru_3/gru_cell/MatMul_1/ReadVariableOpReadVariableOpOmodel_22_gru_3_gru_cell_matmul_1_readvariableop_gru_3_gru_cell_recurrent_kernel*
_output_shapes

: `*
dtype0┤
 model_22/gru_3/gru_cell/MatMul_1MatMulmodel_22/gru_3/zeros:output:07model_22/gru_3/gru_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         `┤
!model_22/gru_3/gru_cell/BiasAdd_1BiasAdd*model_22/gru_3/gru_cell/MatMul_1:product:0(model_22/gru_3/gru_cell/unstack:output:1*
T0*'
_output_shapes
:         `r
model_22/gru_3/gru_cell/ConstConst*
_output_shapes
:*
dtype0*!
valueB"            t
)model_22/gru_3/gru_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         ж
model_22/gru_3/gru_cell/split_1SplitV*model_22/gru_3/gru_cell/BiasAdd_1:output:0&model_22/gru_3/gru_cell/Const:output:02model_22/gru_3/gru_cell/split_1/split_dim:output:0*

Tlen0*
T0*M
_output_shapes;
9:          :          :          *
	num_splitи
model_22/gru_3/gru_cell/addAddV2&model_22/gru_3/gru_cell/split:output:0(model_22/gru_3/gru_cell/split_1:output:0*
T0*'
_output_shapes
:          }
model_22/gru_3/gru_cell/SigmoidSigmoidmodel_22/gru_3/gru_cell/add:z:0*
T0*'
_output_shapes
:          к
model_22/gru_3/gru_cell/add_1AddV2&model_22/gru_3/gru_cell/split:output:1(model_22/gru_3/gru_cell/split_1:output:1*
T0*'
_output_shapes
:          Б
!model_22/gru_3/gru_cell/Sigmoid_1Sigmoid!model_22/gru_3/gru_cell/add_1:z:0*
T0*'
_output_shapes
:          е
model_22/gru_3/gru_cell/mulMul%model_22/gru_3/gru_cell/Sigmoid_1:y:0(model_22/gru_3/gru_cell/split_1:output:2*
T0*'
_output_shapes
:          б
model_22/gru_3/gru_cell/add_2AddV2&model_22/gru_3/gru_cell/split:output:2model_22/gru_3/gru_cell/mul:z:0*
T0*'
_output_shapes
:          y
model_22/gru_3/gru_cell/TanhTanh!model_22/gru_3/gru_cell/add_2:z:0*
T0*'
_output_shapes
:          Ъ
model_22/gru_3/gru_cell/mul_1Mul#model_22/gru_3/gru_cell/Sigmoid:y:0model_22/gru_3/zeros:output:0*
T0*'
_output_shapes
:          b
model_22/gru_3/gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?б
model_22/gru_3/gru_cell/subSub&model_22/gru_3/gru_cell/sub/x:output:0#model_22/gru_3/gru_cell/Sigmoid:y:0*
T0*'
_output_shapes
:          Щ
model_22/gru_3/gru_cell/mul_2Mulmodel_22/gru_3/gru_cell/sub:z:0 model_22/gru_3/gru_cell/Tanh:y:0*
T0*'
_output_shapes
:          Ю
model_22/gru_3/gru_cell/add_3AddV2!model_22/gru_3/gru_cell/mul_1:z:0!model_22/gru_3/gru_cell/mul_2:z:0*
T0*'
_output_shapes
:          }
,model_22/gru_3/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        m
+model_22/gru_3/TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :Є
model_22/gru_3/TensorArrayV2_1TensorListReserve5model_22/gru_3/TensorArrayV2_1/element_shape:output:04model_22/gru_3/TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥U
model_22/gru_3/timeConst*
_output_shapes
: *
dtype0*
value	B : w
,model_22/gru_3/TensorArrayV2_2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         х
model_22/gru_3/TensorArrayV2_2TensorListReserve5model_22/gru_3/TensorArrayV2_2/element_shape:output:0'model_22/gru_3/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0
*

shape_type0:щш╚Ч
Fmodel_22/gru_3/TensorArrayUnstack_1/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       У
8model_22/gru_3/TensorArrayUnstack_1/TensorListFromTensorTensorListFromTensormodel_22/gru_3/transpose_1:y:0Omodel_22/gru_3/TensorArrayUnstack_1/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0
*

shape_type0:щш╚{
model_22/gru_3/zeros_like	ZerosLike!model_22/gru_3/gru_cell/add_3:z:0*
T0*'
_output_shapes
:          r
'model_22/gru_3/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         c
!model_22/gru_3/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Э
model_22/gru_3/whileWhile*model_22/gru_3/while/loop_counter:output:00model_22/gru_3/while/maximum_iterations:output:0model_22/gru_3/time:output:0'model_22/gru_3/TensorArrayV2_1:handle:0model_22/gru_3/zeros_like:y:0model_22/gru_3/zeros:output:0'model_22/gru_3/strided_slice_1:output:0Fmodel_22/gru_3/TensorArrayUnstack/TensorListFromTensor:output_handle:0Hmodel_22/gru_3/TensorArrayUnstack_1/TensorListFromTensor:output_handle:0:model_22_gru_3_gru_cell_readvariableop_gru_3_gru_cell_biasCmodel_22_gru_3_gru_cell_matmul_readvariableop_gru_3_gru_cell_kernelOmodel_22_gru_3_gru_cell_matmul_1_readvariableop_gru_3_gru_cell_recurrent_kernel*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :          :          : : : : : : *%
_read_only_resource_inputs
	
*+
body#R!
model_22_gru_3_while_body_20553*+
cond#R!
model_22_gru_3_while_cond_20552*M
output_shapes<
:: : : : :          :          : : : : : : *
parallel_iterations Р
?model_22/gru_3/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        Г
1model_22/gru_3/TensorArrayV2Stack/TensorListStackTensorListStackmodel_22/gru_3/while:output:3Hmodel_22/gru_3/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:          *
element_dtype0*
num_elementsw
$model_22/gru_3/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         p
&model_22/gru_3/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: p
&model_22/gru_3/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╥
model_22/gru_3/strided_slice_3StridedSlice:model_22/gru_3/TensorArrayV2Stack/TensorListStack:tensor:0-model_22/gru_3/strided_slice_3/stack:output:0/model_22/gru_3/strided_slice_3/stack_1:output:0/model_22/gru_3/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:          *
shrink_axis_maskt
model_22/gru_3/transpose_2/permConst*
_output_shapes
:*
dtype0*!
valueB"          ├
model_22/gru_3/transpose_2	Transpose:model_22/gru_3/TensorArrayV2Stack/TensorListStack:tensor:0(model_22/gru_3/transpose_2/perm:output:0*
T0*+
_output_shapes
:          j
model_22/gru_3/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    Я
'model_22/dense_19/MatMul/ReadVariableOpReadVariableOp7model_22_dense_19_matmul_readvariableop_dense_19_kernel*
_output_shapes

: *
dtype0о
model_22/dense_19/MatMulMatMul'model_22/gru_3/strided_slice_3:output:0/model_22/dense_19/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         Ы
(model_22/dense_19/BiasAdd/ReadVariableOpReadVariableOp6model_22_dense_19_biasadd_readvariableop_dense_19_bias*
_output_shapes
:*
dtype0м
model_22/dense_19/BiasAddBiasAdd"model_22/dense_19/MatMul:product:00model_22/dense_19/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         z
model_22/dense_19/SoftmaxSoftmax"model_22/dense_19/BiasAdd:output:0*
T0*'
_output_shapes
:         r
IdentityIdentity#model_22/dense_19/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:         Щ
NoOpNoOp)^model_22/dense_19/BiasAdd/ReadVariableOp(^model_22/dense_19/MatMul/ReadVariableOp.^model_22/gru_3/gru_cell/MatMul/ReadVariableOp0^model_22/gru_3/gru_cell/MatMul_1/ReadVariableOp'^model_22/gru_3/gru_cell/ReadVariableOp^model_22/gru_3/while*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":         │
: : : : : 2T
(model_22/dense_19/BiasAdd/ReadVariableOp(model_22/dense_19/BiasAdd/ReadVariableOp2R
'model_22/dense_19/MatMul/ReadVariableOp'model_22/dense_19/MatMul/ReadVariableOp2^
-model_22/gru_3/gru_cell/MatMul/ReadVariableOp-model_22/gru_3/gru_cell/MatMul/ReadVariableOp2b
/model_22/gru_3/gru_cell/MatMul_1/ReadVariableOp/model_22/gru_3/gru_cell/MatMul_1/ReadVariableOp2P
&model_22/gru_3/gru_cell/ReadVariableOp&model_22/gru_3/gru_cell/ReadVariableOp2,
model_22/gru_3/whilemodel_22/gru_3/while:-)
'
_user_specified_namedense_19/bias:/+
)
_user_specified_namedense_19/kernel:?;
9
_user_specified_name!gru_3/gru_cell/recurrent_kernel:51
/
_user_specified_namegru_3/gru_cell/kernel:3/
-
_user_specified_namegru_3/gru_cell/bias:Y U
,
_output_shapes
:         │

%
_user_specified_nameinput_prior
┘9
║
while_body_21097
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
strided_slice_1_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0?
-gru_cell_readvariableop_gru_3_gru_cell_bias_0:`H
6gru_cell_matmul_readvariableop_gru_3_gru_cell_kernel_0:`T
Bgru_cell_matmul_1_readvariableop_gru_3_gru_cell_recurrent_kernel_0: `
identity

identity_1

identity_2

identity_3

identity_4
strided_slice_1O
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor=
+gru_cell_readvariableop_gru_3_gru_cell_bias:`F
4gru_cell_matmul_readvariableop_gru_3_gru_cell_kernel:`R
@gru_cell_matmul_1_readvariableop_gru_3_gru_cell_recurrent_kernel: `Ивgru_cell/MatMul/ReadVariableOpв gru_cell/MatMul_1/ReadVariableOpвgru_cell/ReadVariableOpВ
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       О
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0Е
gru_cell/ReadVariableOpReadVariableOp-gru_cell_readvariableop_gru_3_gru_cell_bias_0*
_output_shapes

:`*
dtype0q
gru_cell/unstackUnpackgru_cell/ReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
numХ
gru_cell/MatMul/ReadVariableOpReadVariableOp6gru_cell_matmul_readvariableop_gru_3_gru_cell_kernel_0*
_output_shapes

:`*
dtype0Я
gru_cell/MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0&gru_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         `Г
gru_cell/BiasAddBiasAddgru_cell/MatMul:product:0gru_cell/unstack:output:0*
T0*'
_output_shapes
:         `c
gru_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         ╛
gru_cell/splitSplit!gru_cell/split/split_dim:output:0gru_cell/BiasAdd:output:0*
T0*M
_output_shapes;
9:          :          :          *
	num_splitг
 gru_cell/MatMul_1/ReadVariableOpReadVariableOpBgru_cell_matmul_1_readvariableop_gru_3_gru_cell_recurrent_kernel_0*
_output_shapes

: `*
dtype0Ж
gru_cell/MatMul_1MatMulplaceholder_2(gru_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         `З
gru_cell/BiasAdd_1BiasAddgru_cell/MatMul_1:product:0gru_cell/unstack:output:1*
T0*'
_output_shapes
:         `c
gru_cell/ConstConst*
_output_shapes
:*
dtype0*!
valueB"            e
gru_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         ъ
gru_cell/split_1SplitVgru_cell/BiasAdd_1:output:0gru_cell/Const:output:0#gru_cell/split_1/split_dim:output:0*

Tlen0*
T0*M
_output_shapes;
9:          :          :          *
	num_split{
gru_cell/addAddV2gru_cell/split:output:0gru_cell/split_1:output:0*
T0*'
_output_shapes
:          _
gru_cell/SigmoidSigmoidgru_cell/add:z:0*
T0*'
_output_shapes
:          }
gru_cell/add_1AddV2gru_cell/split:output:1gru_cell/split_1:output:1*
T0*'
_output_shapes
:          c
gru_cell/Sigmoid_1Sigmoidgru_cell/add_1:z:0*
T0*'
_output_shapes
:          x
gru_cell/mulMulgru_cell/Sigmoid_1:y:0gru_cell/split_1:output:2*
T0*'
_output_shapes
:          t
gru_cell/add_2AddV2gru_cell/split:output:2gru_cell/mul:z:0*
T0*'
_output_shapes
:          [
gru_cell/TanhTanhgru_cell/add_2:z:0*
T0*'
_output_shapes
:          l
gru_cell/mul_1Mulgru_cell/Sigmoid:y:0placeholder_2*
T0*'
_output_shapes
:          S
gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?t
gru_cell/subSubgru_cell/sub/x:output:0gru_cell/Sigmoid:y:0*
T0*'
_output_shapes
:          l
gru_cell/mul_2Mulgru_cell/sub:z:0gru_cell/Tanh:y:0*
T0*'
_output_shapes
:          q
gru_cell/add_3AddV2gru_cell/mul_1:z:0gru_cell/mul_2:z:0*
T0*'
_output_shapes
:          l
*TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : ╤
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_13TensorArrayV2Write/TensorListSetItem/index:output:0gru_cell/add_3:z:0*
_output_shapes
: *
element_dtype0:щш╥G
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
:          А
NoOpNoOp^gru_cell/MatMul/ReadVariableOp!^gru_cell/MatMul_1/ReadVariableOp^gru_cell/ReadVariableOp*
_output_shapes
 "Ж
@gru_cell_matmul_1_readvariableop_gru_3_gru_cell_recurrent_kernelBgru_cell_matmul_1_readvariableop_gru_3_gru_cell_recurrent_kernel_0"n
4gru_cell_matmul_readvariableop_gru_3_gru_cell_kernel6gru_cell_matmul_readvariableop_gru_3_gru_cell_kernel_0"\
+gru_cell_readvariableop_gru_3_gru_cell_bias-gru_cell_readvariableop_gru_3_gru_cell_bias_0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
identityIdentity:output:0"$
strided_slice_1strided_slice_1_0"Ь
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :          : : : : : 2@
gru_cell/MatMul/ReadVariableOpgru_cell/MatMul/ReadVariableOp2D
 gru_cell/MatMul_1/ReadVariableOp gru_cell/MatMul_1/ReadVariableOp22
gru_cell/ReadVariableOpgru_cell/ReadVariableOp:?	;
9
_user_specified_name!gru_3/gru_cell/recurrent_kernel:51
/
_user_specified_namegru_3/gru_cell/kernel:3/
-
_user_specified_namegru_3/gru_cell/bias:_[
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
:          :
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
╪
Б
C__inference_gru_cell_layer_call_and_return_conditional_losses_20868

inputs

states4
"readvariableop_gru_3_gru_cell_bias:`=
+matmul_readvariableop_gru_3_gru_cell_kernel:`I
7matmul_1_readvariableop_gru_3_gru_cell_recurrent_kernel: `
identity

identity_1ИвMatMul/ReadVariableOpвMatMul_1/ReadVariableOpвReadVariableOpq
ReadVariableOpReadVariableOp"readvariableop_gru_3_gru_cell_bias*
_output_shapes

:`*
dtype0_
unstackUnpackReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
numБ
MatMul/ReadVariableOpReadVariableOp+matmul_readvariableop_gru_3_gru_cell_kernel*
_output_shapes

:`*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         `h
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*'
_output_shapes
:         `Z
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         г
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*M
_output_shapes;
9:          :          :          *
	num_splitП
MatMul_1/ReadVariableOpReadVariableOp7matmul_1_readvariableop_gru_3_gru_cell_recurrent_kernel*
_output_shapes

: `*
dtype0m
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         `l
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*'
_output_shapes
:         `Z
ConstConst*
_output_shapes
:*
dtype0*!
valueB"            \
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         ╞
split_1SplitVBiasAdd_1:output:0Const:output:0split_1/split_dim:output:0*

Tlen0*
T0*M
_output_shapes;
9:          :          :          *
	num_split`
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:          M
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:          b
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:          Q
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:          ]
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:          Y
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:          I
TanhTanh	add_2:z:0*
T0*'
_output_shapes
:          S
mul_1MulSigmoid:y:0states*
T0*'
_output_shapes
:          J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:          Q
mul_2Mulsub:z:0Tanh:y:0*
T0*'
_output_shapes
:          V
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*'
_output_shapes
:          X
IdentityIdentity	add_3:z:0^NoOp*
T0*'
_output_shapes
:          Z

Identity_1Identity	add_3:z:0^NoOp*
T0*'
_output_shapes
:          e
NoOpNoOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
_output_shapes
 "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:         :          : : : 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp:?;
9
_user_specified_name!gru_3/gru_cell/recurrent_kernel:51
/
_user_specified_namegru_3/gru_cell/kernel:3/
-
_user_specified_namegru_3/gru_cell/bias:OK
'
_output_shapes
:          
 
_user_specified_namestates:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
О
В
(__inference_gru_cell_layer_call_fn_22227

inputs
states_0%
gru_3_gru_cell_bias:`'
gru_3_gru_cell_kernel:`1
gru_3_gru_cell_recurrent_kernel: `
identity

identity_1ИвStatefulPartitionedCall▓
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0gru_3_gru_cell_biasgru_3_gru_cell_kernelgru_3_gru_cell_recurrent_kernel*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:          :          *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_gru_cell_layer_call_and_return_conditional_losses_20730o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:          q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:          <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:         :          : : : 22
StatefulPartitionedCallStatefulPartitionedCall:?;
9
_user_specified_name!gru_3/gru_cell/recurrent_kernel:51
/
_user_specified_namegru_3/gru_cell/kernel:3/
-
_user_specified_namegru_3/gru_cell/bias:QM
'
_output_shapes
:          
"
_user_specified_name
states_0:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
─
l
@__inference_dot_7_layer_call_and_return_conditional_losses_21521
inputs_0
inputs_1
identityb
MatMulBatchMatMulV2inputs_0inputs_1*
T0*,
_output_shapes
:         │
R
ShapeShapeMatMul:output:0*
T0*
_output_shapes
::э╧\
IdentityIdentityMatMul:output:0*
T0*,
_output_shapes
:         │
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:         ││:         │
:VR
,
_output_shapes
:         │

"
_user_specified_name
inputs_1:W S
-
_output_shapes
:         ││
"
_user_specified_name
inputs_0
╩
d
H__inference_masking_prior_layer_call_and_return_conditional_losses_21483

inputs
identityO

NotEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А┐h
NotEqualNotEqualinputsNotEqual/y:output:0*
T0*,
_output_shapes
:         │
`
Any/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
         w
AnyAnyNotEqual:z:0Any/reduction_indices:output:0*,
_output_shapes
:         │*
	keep_dims(`
CastCastAny:output:0*

DstT0*

SrcT0
*,
_output_shapes
:         │S
mulMulinputsCast:y:0*
T0*,
_output_shapes
:         │
s
SqueezeSqueezeAny:output:0*
T0
*(
_output_shapes
:         │*
squeeze_dims

         T
IdentityIdentitymul:z:0*
T0*,
_output_shapes
:         │
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         │
:T P
,
_output_shapes
:         │

 
_user_specified_nameinputs
Ў
у
while_cond_21096
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
less_strided_slice_1-
)while_cond_21096___redundant_placeholder0-
)while_cond_21096___redundant_placeholder1-
)while_cond_21096___redundant_placeholder2-
)while_cond_21096___redundant_placeholder3
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
-: : : : :          : :::::
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
:          :
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
є

А
C__inference_dense_19_layer_call_and_return_conditional_losses_21202

inputs7
%matmul_readvariableop_dense_19_kernel: 2
$biasadd_readvariableop_dense_19_bias:
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOp{
MatMul/ReadVariableOpReadVariableOp%matmul_readvariableop_dense_19_kernel*
_output_shapes

: *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         w
BiasAdd/ReadVariableOpReadVariableOp$biasadd_readvariableop_dense_19_bias*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:         `
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:         S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:          : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:-)
'
_user_specified_namedense_19/bias:/+
)
_user_specified_namedense_19/kernel:O K
'
_output_shapes
:          
 
_user_specified_nameinputs
╓
o
E__inference_multiply_3_layer_call_and_return_conditional_losses_21023

inputs
inputs_1
identityS
mulMulinputsinputs_1*
T0*,
_output_shapes
:         │
T
IdentityIdentitymul:z:0*
T0*,
_output_shapes
:         │
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:         │
:         │
:TP
,
_output_shapes
:         │

 
_user_specified_nameinputs:T P
,
_output_shapes
:         │

 
_user_specified_nameinputs
┘9
║
while_body_21953
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
strided_slice_1_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0?
-gru_cell_readvariableop_gru_3_gru_cell_bias_0:`H
6gru_cell_matmul_readvariableop_gru_3_gru_cell_kernel_0:`T
Bgru_cell_matmul_1_readvariableop_gru_3_gru_cell_recurrent_kernel_0: `
identity

identity_1

identity_2

identity_3

identity_4
strided_slice_1O
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor=
+gru_cell_readvariableop_gru_3_gru_cell_bias:`F
4gru_cell_matmul_readvariableop_gru_3_gru_cell_kernel:`R
@gru_cell_matmul_1_readvariableop_gru_3_gru_cell_recurrent_kernel: `Ивgru_cell/MatMul/ReadVariableOpв gru_cell/MatMul_1/ReadVariableOpвgru_cell/ReadVariableOpВ
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       О
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0Е
gru_cell/ReadVariableOpReadVariableOp-gru_cell_readvariableop_gru_3_gru_cell_bias_0*
_output_shapes

:`*
dtype0q
gru_cell/unstackUnpackgru_cell/ReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
numХ
gru_cell/MatMul/ReadVariableOpReadVariableOp6gru_cell_matmul_readvariableop_gru_3_gru_cell_kernel_0*
_output_shapes

:`*
dtype0Я
gru_cell/MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0&gru_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         `Г
gru_cell/BiasAddBiasAddgru_cell/MatMul:product:0gru_cell/unstack:output:0*
T0*'
_output_shapes
:         `c
gru_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         ╛
gru_cell/splitSplit!gru_cell/split/split_dim:output:0gru_cell/BiasAdd:output:0*
T0*M
_output_shapes;
9:          :          :          *
	num_splitг
 gru_cell/MatMul_1/ReadVariableOpReadVariableOpBgru_cell_matmul_1_readvariableop_gru_3_gru_cell_recurrent_kernel_0*
_output_shapes

: `*
dtype0Ж
gru_cell/MatMul_1MatMulplaceholder_2(gru_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         `З
gru_cell/BiasAdd_1BiasAddgru_cell/MatMul_1:product:0gru_cell/unstack:output:1*
T0*'
_output_shapes
:         `c
gru_cell/ConstConst*
_output_shapes
:*
dtype0*!
valueB"            e
gru_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         ъ
gru_cell/split_1SplitVgru_cell/BiasAdd_1:output:0gru_cell/Const:output:0#gru_cell/split_1/split_dim:output:0*

Tlen0*
T0*M
_output_shapes;
9:          :          :          *
	num_split{
gru_cell/addAddV2gru_cell/split:output:0gru_cell/split_1:output:0*
T0*'
_output_shapes
:          _
gru_cell/SigmoidSigmoidgru_cell/add:z:0*
T0*'
_output_shapes
:          }
gru_cell/add_1AddV2gru_cell/split:output:1gru_cell/split_1:output:1*
T0*'
_output_shapes
:          c
gru_cell/Sigmoid_1Sigmoidgru_cell/add_1:z:0*
T0*'
_output_shapes
:          x
gru_cell/mulMulgru_cell/Sigmoid_1:y:0gru_cell/split_1:output:2*
T0*'
_output_shapes
:          t
gru_cell/add_2AddV2gru_cell/split:output:2gru_cell/mul:z:0*
T0*'
_output_shapes
:          [
gru_cell/TanhTanhgru_cell/add_2:z:0*
T0*'
_output_shapes
:          l
gru_cell/mul_1Mulgru_cell/Sigmoid:y:0placeholder_2*
T0*'
_output_shapes
:          S
gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?t
gru_cell/subSubgru_cell/sub/x:output:0gru_cell/Sigmoid:y:0*
T0*'
_output_shapes
:          l
gru_cell/mul_2Mulgru_cell/sub:z:0gru_cell/Tanh:y:0*
T0*'
_output_shapes
:          q
gru_cell/add_3AddV2gru_cell/mul_1:z:0gru_cell/mul_2:z:0*
T0*'
_output_shapes
:          l
*TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : ╤
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_13TensorArrayV2Write/TensorListSetItem/index:output:0gru_cell/add_3:z:0*
_output_shapes
: *
element_dtype0:щш╥G
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
:          А
NoOpNoOp^gru_cell/MatMul/ReadVariableOp!^gru_cell/MatMul_1/ReadVariableOp^gru_cell/ReadVariableOp*
_output_shapes
 "Ж
@gru_cell_matmul_1_readvariableop_gru_3_gru_cell_recurrent_kernelBgru_cell_matmul_1_readvariableop_gru_3_gru_cell_recurrent_kernel_0"n
4gru_cell_matmul_readvariableop_gru_3_gru_cell_kernel6gru_cell_matmul_readvariableop_gru_3_gru_cell_kernel_0"\
+gru_cell_readvariableop_gru_3_gru_cell_bias-gru_cell_readvariableop_gru_3_gru_cell_bias_0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
identityIdentity:output:0"$
strided_slice_1strided_slice_1_0"Ь
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :          : : : : : 2@
gru_cell/MatMul/ReadVariableOpgru_cell/MatMul/ReadVariableOp2D
 gru_cell/MatMul_1/ReadVariableOp gru_cell/MatMul_1/ReadVariableOp22
gru_cell/ReadVariableOpgru_cell/ReadVariableOp:?	;
9
_user_specified_name!gru_3/gru_cell/recurrent_kernel:51
/
_user_specified_namegru_3/gru_cell/kernel:3/
-
_user_specified_namegru_3/gru_cell/bias:_[
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
:          :
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
ш
c
G__inference_activation_3_layer_call_and_return_conditional_losses_21508

inputs
identityR
SoftmaxSoftmaxinputs*
T0*-
_output_shapes
:         ││_
IdentityIdentitySoftmax:softmax:0*
T0*-
_output_shapes
:         ││"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:         ││:U Q
-
_output_shapes
:         ││
 
_user_specified_nameinputs
║
Q
%__inference_dot_6_layer_call_fn_21489
inputs_0
inputs_1
identity╛
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:         ││* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *I
fDRB
@__inference_dot_6_layer_call_and_return_conditional_losses_21002f
IdentityIdentityPartitionedCall:output:0*
T0*-
_output_shapes
:         ││"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:         │
:         │
:VR
,
_output_shapes
:         │

"
_user_specified_name
inputs_1:V R
,
_output_shapes
:         │

"
_user_specified_name
inputs_0
╥N
в
@__inference_gru_3_layer_call_and_return_conditional_losses_21888
inputs_0=
+gru_cell_readvariableop_gru_3_gru_cell_bias:`F
4gru_cell_matmul_readvariableop_gru_3_gru_cell_kernel:`R
@gru_cell_matmul_1_readvariableop_gru_3_gru_cell_recurrent_kernel: `
identityИвgru_cell/MatMul/ReadVariableOpв gru_cell/MatMul_1/ReadVariableOpвgru_cell/ReadVariableOpвwhileK
ShapeShapeinputs_0*
T0*
_output_shapes
::э╧]
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
valueB:╤
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
:          c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :                  R
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
::э╧_
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
valueB:█
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
         ┤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥Ж
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       р
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥_
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
:         *
shrink_axis_maskГ
gru_cell/ReadVariableOpReadVariableOp+gru_cell_readvariableop_gru_3_gru_cell_bias*
_output_shapes

:`*
dtype0q
gru_cell/unstackUnpackgru_cell/ReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
numУ
gru_cell/MatMul/ReadVariableOpReadVariableOp4gru_cell_matmul_readvariableop_gru_3_gru_cell_kernel*
_output_shapes

:`*
dtype0Н
gru_cell/MatMulMatMulstrided_slice_2:output:0&gru_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         `Г
gru_cell/BiasAddBiasAddgru_cell/MatMul:product:0gru_cell/unstack:output:0*
T0*'
_output_shapes
:         `c
gru_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         ╛
gru_cell/splitSplit!gru_cell/split/split_dim:output:0gru_cell/BiasAdd:output:0*
T0*M
_output_shapes;
9:          :          :          *
	num_splitб
 gru_cell/MatMul_1/ReadVariableOpReadVariableOp@gru_cell_matmul_1_readvariableop_gru_3_gru_cell_recurrent_kernel*
_output_shapes

: `*
dtype0З
gru_cell/MatMul_1MatMulzeros:output:0(gru_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         `З
gru_cell/BiasAdd_1BiasAddgru_cell/MatMul_1:product:0gru_cell/unstack:output:1*
T0*'
_output_shapes
:         `c
gru_cell/ConstConst*
_output_shapes
:*
dtype0*!
valueB"            e
gru_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         ъ
gru_cell/split_1SplitVgru_cell/BiasAdd_1:output:0gru_cell/Const:output:0#gru_cell/split_1/split_dim:output:0*

Tlen0*
T0*M
_output_shapes;
9:          :          :          *
	num_split{
gru_cell/addAddV2gru_cell/split:output:0gru_cell/split_1:output:0*
T0*'
_output_shapes
:          _
gru_cell/SigmoidSigmoidgru_cell/add:z:0*
T0*'
_output_shapes
:          }
gru_cell/add_1AddV2gru_cell/split:output:1gru_cell/split_1:output:1*
T0*'
_output_shapes
:          c
gru_cell/Sigmoid_1Sigmoidgru_cell/add_1:z:0*
T0*'
_output_shapes
:          x
gru_cell/mulMulgru_cell/Sigmoid_1:y:0gru_cell/split_1:output:2*
T0*'
_output_shapes
:          t
gru_cell/add_2AddV2gru_cell/split:output:2gru_cell/mul:z:0*
T0*'
_output_shapes
:          [
gru_cell/TanhTanhgru_cell/add_2:z:0*
T0*'
_output_shapes
:          m
gru_cell/mul_1Mulgru_cell/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:          S
gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?t
gru_cell/subSubgru_cell/sub/x:output:0gru_cell/Sigmoid:y:0*
T0*'
_output_shapes
:          l
gru_cell/mul_2Mulgru_cell/sub:z:0gru_cell/Tanh:y:0*
T0*'
_output_shapes
:          q
gru_cell/add_3AddV2gru_cell/mul_1:z:0gru_cell/mul_2:z:0*
T0*'
_output_shapes
:          n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :┼
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥F
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
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ┼
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+gru_cell_readvariableop_gru_3_gru_cell_bias4gru_cell_matmul_readvariableop_gru_3_gru_cell_kernel@gru_cell_matmul_1_readvariableop_gru_3_gru_cell_recurrent_kernel*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :          : : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_21798*
condR
while_cond_21797*8
output_shapes'
%: : : : :          : : : : : *
parallel_iterations Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        ╓
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:          *
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:З
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:          *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ц
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:          [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:          И
NoOpNoOp^gru_cell/MatMul/ReadVariableOp!^gru_cell/MatMul_1/ReadVariableOp^gru_cell/ReadVariableOp^while*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  : : : 2@
gru_cell/MatMul/ReadVariableOpgru_cell/MatMul/ReadVariableOp2D
 gru_cell/MatMul_1/ReadVariableOp gru_cell/MatMul_1/ReadVariableOp22
gru_cell/ReadVariableOpgru_cell/ReadVariableOp2
whilewhile:?;
9
_user_specified_name!gru_3/gru_cell/recurrent_kernel:51
/
_user_specified_namegru_3/gru_cell/kernel:3/
-
_user_specified_namegru_3/gru_cell/bias:^ Z
4
_output_shapes"
 :                  
"
_user_specified_name
inputs_0
╚
Y
-__inference_concatenate_3_layer_call_fn_21539
inputs_0
inputs_1
identity┼
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         │* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_concatenate_3_layer_call_and_return_conditional_losses_21031e
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:         │"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:         │
:         │
:VR
,
_output_shapes
:         │

"
_user_specified_name
inputs_1:V R
,
_output_shapes
:         │

"
_user_specified_name
inputs_0
╗
H
,__inference_activation_3_layer_call_fn_21503

inputs
identity╕
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:         ││* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_activation_3_layer_call_and_return_conditional_losses_21008f
IdentityIdentityPartitionedCall:output:0*
T0*-
_output_shapes
:         ││"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:         ││:U Q
-
_output_shapes
:         ││
 
_user_specified_nameinputs
Ш
j
@__inference_dot_6_layer_call_and_return_conditional_losses_21002

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
:         
│f
MatMulBatchMatMulV2inputstranspose:y:0*
T0*-
_output_shapes
:         ││R
ShapeShapeMatMul:output:0*
T0*
_output_shapes
::э╧]
IdentityIdentityMatMul:output:0*
T0*-
_output_shapes
:         ││"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:         │
:         │
:TP
,
_output_shapes
:         │

 
_user_specified_nameinputs:T P
,
_output_shapes
:         │

 
_user_specified_nameinputs
╙А
╦
!__inference__traced_restore_22581
file_prefix2
 assignvariableop_dense_19_kernel: .
 assignvariableop_1_dense_19_bias::
(assignvariableop_2_gru_3_gru_cell_kernel:`D
2assignvariableop_3_gru_3_gru_cell_recurrent_kernel: `8
&assignvariableop_4_gru_3_gru_cell_bias:`1
'assignvariableop_5_training_6_adam_iter:	 3
)assignvariableop_6_training_6_adam_beta_1: 3
)assignvariableop_7_training_6_adam_beta_2: 2
(assignvariableop_8_training_6_adam_decay: :
0assignvariableop_9_training_6_adam_learning_rate: %
assignvariableop_10_total_3: %
assignvariableop_11_count_3: 3
$assignvariableop_12_true_positives_3:	╚3
$assignvariableop_13_true_negatives_3:	╚4
%assignvariableop_14_false_positives_3:	╚4
%assignvariableop_15_false_negatives_3:	╚G
5assignvariableop_16_training_6_adam_dense_19_kernel_m: A
3assignvariableop_17_training_6_adam_dense_19_bias_m:M
;assignvariableop_18_training_6_adam_gru_3_gru_cell_kernel_m:`W
Eassignvariableop_19_training_6_adam_gru_3_gru_cell_recurrent_kernel_m: `K
9assignvariableop_20_training_6_adam_gru_3_gru_cell_bias_m:`G
5assignvariableop_21_training_6_adam_dense_19_kernel_v: A
3assignvariableop_22_training_6_adam_dense_19_bias_v:M
;assignvariableop_23_training_6_adam_gru_3_gru_cell_kernel_v:`W
Eassignvariableop_24_training_6_adam_gru_3_gru_cell_recurrent_kernel_v: `K
9assignvariableop_25_training_6_adam_gru_3_gru_cell_bias_v:`
identity_27ИвAssignVariableOpвAssignVariableOp_1вAssignVariableOp_10вAssignVariableOp_11вAssignVariableOp_12вAssignVariableOp_13вAssignVariableOp_14вAssignVariableOp_15вAssignVariableOp_16вAssignVariableOp_17вAssignVariableOp_18вAssignVariableOp_19вAssignVariableOp_2вAssignVariableOp_20вAssignVariableOp_21вAssignVariableOp_22вAssignVariableOp_23вAssignVariableOp_24вAssignVariableOp_25вAssignVariableOp_3вAssignVariableOp_4вAssignVariableOp_5вAssignVariableOp_6вAssignVariableOp_7вAssignVariableOp_8вAssignVariableOp_9ж
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*╠
value┬B┐B6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/1/true_positives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/1/true_negatives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/1/false_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/1/false_negatives/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHж
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*I
value@B>B B B B B B B B B B B B B B B B B B B B B B B B B B B ж
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*А
_output_shapesn
l:::::::::::::::::::::::::::*)
dtypes
2	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:│
AssignVariableOpAssignVariableOp assignvariableop_dense_19_kernelIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:╖
AssignVariableOp_1AssignVariableOp assignvariableop_1_dense_19_biasIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:┐
AssignVariableOp_2AssignVariableOp(assignvariableop_2_gru_3_gru_cell_kernelIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:╔
AssignVariableOp_3AssignVariableOp2assignvariableop_3_gru_3_gru_cell_recurrent_kernelIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:╜
AssignVariableOp_4AssignVariableOp&assignvariableop_4_gru_3_gru_cell_biasIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0	*
_output_shapes
:╛
AssignVariableOp_5AssignVariableOp'assignvariableop_5_training_6_adam_iterIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:└
AssignVariableOp_6AssignVariableOp)assignvariableop_6_training_6_adam_beta_1Identity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:└
AssignVariableOp_7AssignVariableOp)assignvariableop_7_training_6_adam_beta_2Identity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:┐
AssignVariableOp_8AssignVariableOp(assignvariableop_8_training_6_adam_decayIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:╟
AssignVariableOp_9AssignVariableOp0assignvariableop_9_training_6_adam_learning_rateIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:┤
AssignVariableOp_10AssignVariableOpassignvariableop_10_total_3Identity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:┤
AssignVariableOp_11AssignVariableOpassignvariableop_11_count_3Identity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:╜
AssignVariableOp_12AssignVariableOp$assignvariableop_12_true_positives_3Identity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:╜
AssignVariableOp_13AssignVariableOp$assignvariableop_13_true_negatives_3Identity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:╛
AssignVariableOp_14AssignVariableOp%assignvariableop_14_false_positives_3Identity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:╛
AssignVariableOp_15AssignVariableOp%assignvariableop_15_false_negatives_3Identity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:╬
AssignVariableOp_16AssignVariableOp5assignvariableop_16_training_6_adam_dense_19_kernel_mIdentity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:╠
AssignVariableOp_17AssignVariableOp3assignvariableop_17_training_6_adam_dense_19_bias_mIdentity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:╘
AssignVariableOp_18AssignVariableOp;assignvariableop_18_training_6_adam_gru_3_gru_cell_kernel_mIdentity_18:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:▐
AssignVariableOp_19AssignVariableOpEassignvariableop_19_training_6_adam_gru_3_gru_cell_recurrent_kernel_mIdentity_19:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:╥
AssignVariableOp_20AssignVariableOp9assignvariableop_20_training_6_adam_gru_3_gru_cell_bias_mIdentity_20:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:╬
AssignVariableOp_21AssignVariableOp5assignvariableop_21_training_6_adam_dense_19_kernel_vIdentity_21:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:╠
AssignVariableOp_22AssignVariableOp3assignvariableop_22_training_6_adam_dense_19_bias_vIdentity_22:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:╘
AssignVariableOp_23AssignVariableOp;assignvariableop_23_training_6_adam_gru_3_gru_cell_kernel_vIdentity_23:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:▐
AssignVariableOp_24AssignVariableOpEassignvariableop_24_training_6_adam_gru_3_gru_cell_recurrent_kernel_vIdentity_24:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:╥
AssignVariableOp_25AssignVariableOp9assignvariableop_25_training_6_adam_gru_3_gru_cell_bias_vIdentity_25:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 Л
Identity_26Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_27IdentityIdentity_26:output:0^NoOp_1*
T0*
_output_shapes
: ╘
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
AssignVariableOpAssignVariableOp:EA
?
_user_specified_name'%training_6/Adam/gru_3/gru_cell/bias/v:QM
K
_user_specified_name31training_6/Adam/gru_3/gru_cell/recurrent_kernel/v:GC
A
_user_specified_name)'training_6/Adam/gru_3/gru_cell/kernel/v:?;
9
_user_specified_name!training_6/Adam/dense_19/bias/v:A=
;
_user_specified_name#!training_6/Adam/dense_19/kernel/v:EA
?
_user_specified_name'%training_6/Adam/gru_3/gru_cell/bias/m:QM
K
_user_specified_name31training_6/Adam/gru_3/gru_cell/recurrent_kernel/m:GC
A
_user_specified_name)'training_6/Adam/gru_3/gru_cell/kernel/m:?;
9
_user_specified_name!training_6/Adam/dense_19/bias/m:A=
;
_user_specified_name#!training_6/Adam/dense_19/kernel/m:1-
+
_user_specified_namefalse_negatives_3:1-
+
_user_specified_namefalse_positives_3:0,
*
_user_specified_nametrue_negatives_3:0,
*
_user_specified_nametrue_positives_3:'#
!
_user_specified_name	count_3:'#
!
_user_specified_name	total_3:=
9
7
_user_specified_nametraining_6/Adam/learning_rate:5	1
/
_user_specified_nametraining_6/Adam/decay:62
0
_user_specified_nametraining_6/Adam/beta_2:62
0
_user_specified_nametraining_6/Adam/beta_1:40
.
_user_specified_nametraining_6/Adam/iter:3/
-
_user_specified_namegru_3/gru_cell/bias:?;
9
_user_specified_name!gru_3/gru_cell/recurrent_kernel:51
/
_user_specified_namegru_3/gru_cell/kernel:-)
'
_user_specified_namedense_19/bias:/+
)
_user_specified_namedense_19/kernel:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
▀
t
H__inference_concatenate_3_layer_call_and_return_conditional_losses_21546
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
:         │\
IdentityIdentityconcat:output:0*
T0*,
_output_shapes
:         │"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:         │
:         │
:VR
,
_output_shapes
:         │

"
_user_specified_name
inputs_1:V R
,
_output_shapes
:         │

"
_user_specified_name
inputs_0
р
Г
C__inference_gru_cell_layer_call_and_return_conditional_losses_22316

inputs
states_04
"readvariableop_gru_3_gru_cell_bias:`=
+matmul_readvariableop_gru_3_gru_cell_kernel:`I
7matmul_1_readvariableop_gru_3_gru_cell_recurrent_kernel: `
identity

identity_1ИвMatMul/ReadVariableOpвMatMul_1/ReadVariableOpвReadVariableOpq
ReadVariableOpReadVariableOp"readvariableop_gru_3_gru_cell_bias*
_output_shapes

:`*
dtype0_
unstackUnpackReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
numБ
MatMul/ReadVariableOpReadVariableOp+matmul_readvariableop_gru_3_gru_cell_kernel*
_output_shapes

:`*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         `h
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*'
_output_shapes
:         `Z
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         г
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*M
_output_shapes;
9:          :          :          *
	num_splitП
MatMul_1/ReadVariableOpReadVariableOp7matmul_1_readvariableop_gru_3_gru_cell_recurrent_kernel*
_output_shapes

: `*
dtype0o
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         `l
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*'
_output_shapes
:         `Z
ConstConst*
_output_shapes
:*
dtype0*!
valueB"            \
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         ╞
split_1SplitVBiasAdd_1:output:0Const:output:0split_1/split_dim:output:0*

Tlen0*
T0*M
_output_shapes;
9:          :          :          *
	num_split`
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:          M
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:          b
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:          Q
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:          ]
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:          Y
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:          I
TanhTanh	add_2:z:0*
T0*'
_output_shapes
:          U
mul_1MulSigmoid:y:0states_0*
T0*'
_output_shapes
:          J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:          Q
mul_2Mulsub:z:0Tanh:y:0*
T0*'
_output_shapes
:          V
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*'
_output_shapes
:          X
IdentityIdentity	add_3:z:0^NoOp*
T0*'
_output_shapes
:          Z

Identity_1Identity	add_3:z:0^NoOp*
T0*'
_output_shapes
:          e
NoOpNoOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
_output_shapes
 "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:         :          : : : 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp:?;
9
_user_specified_name!gru_3/gru_cell/recurrent_kernel:51
/
_user_specified_namegru_3/gru_cell/kernel:3/
-
_user_specified_namegru_3/gru_cell/bias:QM
'
_output_shapes
:          
"
_user_specified_name
states_0:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
Ф
б
(__inference_dense_19_layer_call_fn_22205

inputs!
dense_19_kernel: 
dense_19_bias:
identityИвStatefulPartitionedCallф
StatefulPartitionedCallStatefulPartitionedCallinputsdense_19_kerneldense_19_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_dense_19_layer_call_and_return_conditional_losses_21202o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:          : : 22
StatefulPartitionedCallStatefulPartitionedCall:-)
'
_user_specified_namedense_19/bias:/+
)
_user_specified_namedense_19/kernel:O K
'
_output_shapes
:          
 
_user_specified_nameinputs
╣
I
-__inference_masking_prior_layer_call_fn_21472

inputs
identity╕
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         │
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_masking_prior_layer_call_and_return_conditional_losses_20992e
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:         │
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         │
:T P
,
_output_shapes
:         │

 
_user_specified_nameinputs
Ў
у
while_cond_21952
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
less_strided_slice_1-
)while_cond_21952___redundant_placeholder0-
)while_cond_21952___redundant_placeholder1-
)while_cond_21952___redundant_placeholder2-
)while_cond_21952___redundant_placeholder3
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
-: : : : :          : :::::
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
:          :
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
╥N
в
@__inference_gru_3_layer_call_and_return_conditional_losses_21733
inputs_0=
+gru_cell_readvariableop_gru_3_gru_cell_bias:`F
4gru_cell_matmul_readvariableop_gru_3_gru_cell_kernel:`R
@gru_cell_matmul_1_readvariableop_gru_3_gru_cell_recurrent_kernel: `
identityИвgru_cell/MatMul/ReadVariableOpв gru_cell/MatMul_1/ReadVariableOpвgru_cell/ReadVariableOpвwhileK
ShapeShapeinputs_0*
T0*
_output_shapes
::э╧]
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
valueB:╤
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
:          c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :                  R
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
::э╧_
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
valueB:█
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
         ┤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥Ж
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       р
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥_
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
:         *
shrink_axis_maskГ
gru_cell/ReadVariableOpReadVariableOp+gru_cell_readvariableop_gru_3_gru_cell_bias*
_output_shapes

:`*
dtype0q
gru_cell/unstackUnpackgru_cell/ReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
numУ
gru_cell/MatMul/ReadVariableOpReadVariableOp4gru_cell_matmul_readvariableop_gru_3_gru_cell_kernel*
_output_shapes

:`*
dtype0Н
gru_cell/MatMulMatMulstrided_slice_2:output:0&gru_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         `Г
gru_cell/BiasAddBiasAddgru_cell/MatMul:product:0gru_cell/unstack:output:0*
T0*'
_output_shapes
:         `c
gru_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         ╛
gru_cell/splitSplit!gru_cell/split/split_dim:output:0gru_cell/BiasAdd:output:0*
T0*M
_output_shapes;
9:          :          :          *
	num_splitб
 gru_cell/MatMul_1/ReadVariableOpReadVariableOp@gru_cell_matmul_1_readvariableop_gru_3_gru_cell_recurrent_kernel*
_output_shapes

: `*
dtype0З
gru_cell/MatMul_1MatMulzeros:output:0(gru_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         `З
gru_cell/BiasAdd_1BiasAddgru_cell/MatMul_1:product:0gru_cell/unstack:output:1*
T0*'
_output_shapes
:         `c
gru_cell/ConstConst*
_output_shapes
:*
dtype0*!
valueB"            e
gru_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         ъ
gru_cell/split_1SplitVgru_cell/BiasAdd_1:output:0gru_cell/Const:output:0#gru_cell/split_1/split_dim:output:0*

Tlen0*
T0*M
_output_shapes;
9:          :          :          *
	num_split{
gru_cell/addAddV2gru_cell/split:output:0gru_cell/split_1:output:0*
T0*'
_output_shapes
:          _
gru_cell/SigmoidSigmoidgru_cell/add:z:0*
T0*'
_output_shapes
:          }
gru_cell/add_1AddV2gru_cell/split:output:1gru_cell/split_1:output:1*
T0*'
_output_shapes
:          c
gru_cell/Sigmoid_1Sigmoidgru_cell/add_1:z:0*
T0*'
_output_shapes
:          x
gru_cell/mulMulgru_cell/Sigmoid_1:y:0gru_cell/split_1:output:2*
T0*'
_output_shapes
:          t
gru_cell/add_2AddV2gru_cell/split:output:2gru_cell/mul:z:0*
T0*'
_output_shapes
:          [
gru_cell/TanhTanhgru_cell/add_2:z:0*
T0*'
_output_shapes
:          m
gru_cell/mul_1Mulgru_cell/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:          S
gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?t
gru_cell/subSubgru_cell/sub/x:output:0gru_cell/Sigmoid:y:0*
T0*'
_output_shapes
:          l
gru_cell/mul_2Mulgru_cell/sub:z:0gru_cell/Tanh:y:0*
T0*'
_output_shapes
:          q
gru_cell/add_3AddV2gru_cell/mul_1:z:0gru_cell/mul_2:z:0*
T0*'
_output_shapes
:          n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :┼
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥F
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
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ┼
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+gru_cell_readvariableop_gru_3_gru_cell_bias4gru_cell_matmul_readvariableop_gru_3_gru_cell_kernel@gru_cell_matmul_1_readvariableop_gru_3_gru_cell_recurrent_kernel*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :          : : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_21643*
condR
while_cond_21642*8
output_shapes'
%: : : : :          : : : : : *
parallel_iterations Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        ╓
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:          *
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:З
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:          *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ц
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:          [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:          И
NoOpNoOp^gru_cell/MatMul/ReadVariableOp!^gru_cell/MatMul_1/ReadVariableOp^gru_cell/ReadVariableOp^while*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  : : : 2@
gru_cell/MatMul/ReadVariableOpgru_cell/MatMul/ReadVariableOp2D
 gru_cell/MatMul_1/ReadVariableOp gru_cell/MatMul_1/ReadVariableOp22
gru_cell/ReadVariableOpgru_cell/ReadVariableOp2
whilewhile:?;
9
_user_specified_name!gru_3/gru_cell/recurrent_kernel:51
/
_user_specified_namegru_3/gru_cell/kernel:3/
-
_user_specified_namegru_3/gru_cell/bias:^ Z
4
_output_shapes"
 :                  
"
_user_specified_name
inputs_0
Ў
у
while_cond_21797
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
less_strided_slice_1-
)while_cond_21797___redundant_placeholder0-
)while_cond_21797___redundant_placeholder1-
)while_cond_21797___redundant_placeholder2-
)while_cond_21797___redundant_placeholder3
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
-: : : : :          : :::::
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
:          :
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
%__inference_gru_3_layer_call_fn_21554
inputs_0%
gru_3_gru_cell_bias:`'
gru_3_gru_cell_kernel:`1
gru_3_gru_cell_recurrent_kernel: `
identityИвStatefulPartitionedCallТ
StatefulPartitionedCallStatefulPartitionedCallinputs_0gru_3_gru_cell_biasgru_3_gru_cell_kernelgru_3_gru_cell_recurrent_kernel*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:          *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *I
fDRB
@__inference_gru_3_layer_call_and_return_conditional_losses_20803o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:          <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  : : : 22
StatefulPartitionedCallStatefulPartitionedCall:?;
9
_user_specified_name!gru_3/gru_cell/recurrent_kernel:51
/
_user_specified_namegru_3/gru_cell/kernel:3/
-
_user_specified_namegru_3/gru_cell/bias:^ Z
4
_output_shapes"
 :                  
"
_user_specified_name
inputs_0
з╙
е
__inference__traced_save_22494
file_prefix8
&read_disablecopyonread_dense_19_kernel: 4
&read_1_disablecopyonread_dense_19_bias:@
.read_2_disablecopyonread_gru_3_gru_cell_kernel:`J
8read_3_disablecopyonread_gru_3_gru_cell_recurrent_kernel: `>
,read_4_disablecopyonread_gru_3_gru_cell_bias:`7
-read_5_disablecopyonread_training_6_adam_iter:	 9
/read_6_disablecopyonread_training_6_adam_beta_1: 9
/read_7_disablecopyonread_training_6_adam_beta_2: 8
.read_8_disablecopyonread_training_6_adam_decay: @
6read_9_disablecopyonread_training_6_adam_learning_rate: +
!read_10_disablecopyonread_total_3: +
!read_11_disablecopyonread_count_3: 9
*read_12_disablecopyonread_true_positives_3:	╚9
*read_13_disablecopyonread_true_negatives_3:	╚:
+read_14_disablecopyonread_false_positives_3:	╚:
+read_15_disablecopyonread_false_negatives_3:	╚M
;read_16_disablecopyonread_training_6_adam_dense_19_kernel_m: G
9read_17_disablecopyonread_training_6_adam_dense_19_bias_m:S
Aread_18_disablecopyonread_training_6_adam_gru_3_gru_cell_kernel_m:`]
Kread_19_disablecopyonread_training_6_adam_gru_3_gru_cell_recurrent_kernel_m: `Q
?read_20_disablecopyonread_training_6_adam_gru_3_gru_cell_bias_m:`M
;read_21_disablecopyonread_training_6_adam_dense_19_kernel_v: G
9read_22_disablecopyonread_training_6_adam_dense_19_bias_v:S
Aread_23_disablecopyonread_training_6_adam_gru_3_gru_cell_kernel_v:`]
Kread_24_disablecopyonread_training_6_adam_gru_3_gru_cell_recurrent_kernel_v: `Q
?read_25_disablecopyonread_training_6_adam_gru_3_gru_cell_bias_v:`
savev2_const
identity_53ИвMergeV2CheckpointsвRead/DisableCopyOnReadвRead/ReadVariableOpвRead_1/DisableCopyOnReadвRead_1/ReadVariableOpвRead_10/DisableCopyOnReadвRead_10/ReadVariableOpвRead_11/DisableCopyOnReadвRead_11/ReadVariableOpвRead_12/DisableCopyOnReadвRead_12/ReadVariableOpвRead_13/DisableCopyOnReadвRead_13/ReadVariableOpвRead_14/DisableCopyOnReadвRead_14/ReadVariableOpвRead_15/DisableCopyOnReadвRead_15/ReadVariableOpвRead_16/DisableCopyOnReadвRead_16/ReadVariableOpвRead_17/DisableCopyOnReadвRead_17/ReadVariableOpвRead_18/DisableCopyOnReadвRead_18/ReadVariableOpвRead_19/DisableCopyOnReadвRead_19/ReadVariableOpвRead_2/DisableCopyOnReadвRead_2/ReadVariableOpвRead_20/DisableCopyOnReadвRead_20/ReadVariableOpвRead_21/DisableCopyOnReadвRead_21/ReadVariableOpвRead_22/DisableCopyOnReadвRead_22/ReadVariableOpвRead_23/DisableCopyOnReadвRead_23/ReadVariableOpвRead_24/DisableCopyOnReadвRead_24/ReadVariableOpвRead_25/DisableCopyOnReadвRead_25/ReadVariableOpвRead_3/DisableCopyOnReadвRead_3/ReadVariableOpвRead_4/DisableCopyOnReadвRead_4/ReadVariableOpвRead_5/DisableCopyOnReadвRead_5/ReadVariableOpвRead_6/DisableCopyOnReadвRead_6/ReadVariableOpвRead_7/DisableCopyOnReadвRead_7/ReadVariableOpвRead_8/DisableCopyOnReadвRead_8/ReadVariableOpвRead_9/DisableCopyOnReadвRead_9/ReadVariableOpw
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
_temp/partБ
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
value	B : У
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: x
Read/DisableCopyOnReadDisableCopyOnRead&read_disablecopyonread_dense_19_kernel"/device:CPU:0*
_output_shapes
 в
Read/ReadVariableOpReadVariableOp&read_disablecopyonread_dense_19_kernel^Read/DisableCopyOnRead"/device:CPU:0*
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
Read_1/DisableCopyOnReadDisableCopyOnRead&read_1_disablecopyonread_dense_19_bias"/device:CPU:0*
_output_shapes
 в
Read_1/ReadVariableOpReadVariableOp&read_1_disablecopyonread_dense_19_bias^Read_1/DisableCopyOnRead"/device:CPU:0*
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
:В
Read_2/DisableCopyOnReadDisableCopyOnRead.read_2_disablecopyonread_gru_3_gru_cell_kernel"/device:CPU:0*
_output_shapes
 о
Read_2/ReadVariableOpReadVariableOp.read_2_disablecopyonread_gru_3_gru_cell_kernel^Read_2/DisableCopyOnRead"/device:CPU:0*
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

:`М
Read_3/DisableCopyOnReadDisableCopyOnRead8read_3_disablecopyonread_gru_3_gru_cell_recurrent_kernel"/device:CPU:0*
_output_shapes
 ╕
Read_3/ReadVariableOpReadVariableOp8read_3_disablecopyonread_gru_3_gru_cell_recurrent_kernel^Read_3/DisableCopyOnRead"/device:CPU:0*
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

: `А
Read_4/DisableCopyOnReadDisableCopyOnRead,read_4_disablecopyonread_gru_3_gru_cell_bias"/device:CPU:0*
_output_shapes
 м
Read_4/ReadVariableOpReadVariableOp,read_4_disablecopyonread_gru_3_gru_cell_bias^Read_4/DisableCopyOnRead"/device:CPU:0*
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

:`Б
Read_5/DisableCopyOnReadDisableCopyOnRead-read_5_disablecopyonread_training_6_adam_iter"/device:CPU:0*
_output_shapes
 е
Read_5/ReadVariableOpReadVariableOp-read_5_disablecopyonread_training_6_adam_iter^Read_5/DisableCopyOnRead"/device:CPU:0*
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
: Г
Read_6/DisableCopyOnReadDisableCopyOnRead/read_6_disablecopyonread_training_6_adam_beta_1"/device:CPU:0*
_output_shapes
 з
Read_6/ReadVariableOpReadVariableOp/read_6_disablecopyonread_training_6_adam_beta_1^Read_6/DisableCopyOnRead"/device:CPU:0*
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
: Г
Read_7/DisableCopyOnReadDisableCopyOnRead/read_7_disablecopyonread_training_6_adam_beta_2"/device:CPU:0*
_output_shapes
 з
Read_7/ReadVariableOpReadVariableOp/read_7_disablecopyonread_training_6_adam_beta_2^Read_7/DisableCopyOnRead"/device:CPU:0*
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
: В
Read_8/DisableCopyOnReadDisableCopyOnRead.read_8_disablecopyonread_training_6_adam_decay"/device:CPU:0*
_output_shapes
 ж
Read_8/ReadVariableOpReadVariableOp.read_8_disablecopyonread_training_6_adam_decay^Read_8/DisableCopyOnRead"/device:CPU:0*
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
: К
Read_9/DisableCopyOnReadDisableCopyOnRead6read_9_disablecopyonread_training_6_adam_learning_rate"/device:CPU:0*
_output_shapes
 о
Read_9/ReadVariableOpReadVariableOp6read_9_disablecopyonread_training_6_adam_learning_rate^Read_9/DisableCopyOnRead"/device:CPU:0*
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
Read_10/DisableCopyOnReadDisableCopyOnRead!read_10_disablecopyonread_total_3"/device:CPU:0*
_output_shapes
 Ы
Read_10/ReadVariableOpReadVariableOp!read_10_disablecopyonread_total_3^Read_10/DisableCopyOnRead"/device:CPU:0*
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
Read_11/DisableCopyOnReadDisableCopyOnRead!read_11_disablecopyonread_count_3"/device:CPU:0*
_output_shapes
 Ы
Read_11/ReadVariableOpReadVariableOp!read_11_disablecopyonread_count_3^Read_11/DisableCopyOnRead"/device:CPU:0*
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
Read_12/DisableCopyOnReadDisableCopyOnRead*read_12_disablecopyonread_true_positives_3"/device:CPU:0*
_output_shapes
 й
Read_12/ReadVariableOpReadVariableOp*read_12_disablecopyonread_true_positives_3^Read_12/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:╚*
dtype0l
Identity_24IdentityRead_12/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:╚b
Identity_25IdentityIdentity_24:output:0"/device:CPU:0*
T0*
_output_shapes	
:╚
Read_13/DisableCopyOnReadDisableCopyOnRead*read_13_disablecopyonread_true_negatives_3"/device:CPU:0*
_output_shapes
 й
Read_13/ReadVariableOpReadVariableOp*read_13_disablecopyonread_true_negatives_3^Read_13/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:╚*
dtype0l
Identity_26IdentityRead_13/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:╚b
Identity_27IdentityIdentity_26:output:0"/device:CPU:0*
T0*
_output_shapes	
:╚А
Read_14/DisableCopyOnReadDisableCopyOnRead+read_14_disablecopyonread_false_positives_3"/device:CPU:0*
_output_shapes
 к
Read_14/ReadVariableOpReadVariableOp+read_14_disablecopyonread_false_positives_3^Read_14/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:╚*
dtype0l
Identity_28IdentityRead_14/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:╚b
Identity_29IdentityIdentity_28:output:0"/device:CPU:0*
T0*
_output_shapes	
:╚А
Read_15/DisableCopyOnReadDisableCopyOnRead+read_15_disablecopyonread_false_negatives_3"/device:CPU:0*
_output_shapes
 к
Read_15/ReadVariableOpReadVariableOp+read_15_disablecopyonread_false_negatives_3^Read_15/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:╚*
dtype0l
Identity_30IdentityRead_15/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:╚b
Identity_31IdentityIdentity_30:output:0"/device:CPU:0*
T0*
_output_shapes	
:╚Р
Read_16/DisableCopyOnReadDisableCopyOnRead;read_16_disablecopyonread_training_6_adam_dense_19_kernel_m"/device:CPU:0*
_output_shapes
 ╜
Read_16/ReadVariableOpReadVariableOp;read_16_disablecopyonread_training_6_adam_dense_19_kernel_m^Read_16/DisableCopyOnRead"/device:CPU:0*
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

: О
Read_17/DisableCopyOnReadDisableCopyOnRead9read_17_disablecopyonread_training_6_adam_dense_19_bias_m"/device:CPU:0*
_output_shapes
 ╖
Read_17/ReadVariableOpReadVariableOp9read_17_disablecopyonread_training_6_adam_dense_19_bias_m^Read_17/DisableCopyOnRead"/device:CPU:0*
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
:Ц
Read_18/DisableCopyOnReadDisableCopyOnReadAread_18_disablecopyonread_training_6_adam_gru_3_gru_cell_kernel_m"/device:CPU:0*
_output_shapes
 ├
Read_18/ReadVariableOpReadVariableOpAread_18_disablecopyonread_training_6_adam_gru_3_gru_cell_kernel_m^Read_18/DisableCopyOnRead"/device:CPU:0*
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

:`а
Read_19/DisableCopyOnReadDisableCopyOnReadKread_19_disablecopyonread_training_6_adam_gru_3_gru_cell_recurrent_kernel_m"/device:CPU:0*
_output_shapes
 ═
Read_19/ReadVariableOpReadVariableOpKread_19_disablecopyonread_training_6_adam_gru_3_gru_cell_recurrent_kernel_m^Read_19/DisableCopyOnRead"/device:CPU:0*
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

: `Ф
Read_20/DisableCopyOnReadDisableCopyOnRead?read_20_disablecopyonread_training_6_adam_gru_3_gru_cell_bias_m"/device:CPU:0*
_output_shapes
 ┴
Read_20/ReadVariableOpReadVariableOp?read_20_disablecopyonread_training_6_adam_gru_3_gru_cell_bias_m^Read_20/DisableCopyOnRead"/device:CPU:0*
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

:`Р
Read_21/DisableCopyOnReadDisableCopyOnRead;read_21_disablecopyonread_training_6_adam_dense_19_kernel_v"/device:CPU:0*
_output_shapes
 ╜
Read_21/ReadVariableOpReadVariableOp;read_21_disablecopyonread_training_6_adam_dense_19_kernel_v^Read_21/DisableCopyOnRead"/device:CPU:0*
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

: О
Read_22/DisableCopyOnReadDisableCopyOnRead9read_22_disablecopyonread_training_6_adam_dense_19_bias_v"/device:CPU:0*
_output_shapes
 ╖
Read_22/ReadVariableOpReadVariableOp9read_22_disablecopyonread_training_6_adam_dense_19_bias_v^Read_22/DisableCopyOnRead"/device:CPU:0*
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
:Ц
Read_23/DisableCopyOnReadDisableCopyOnReadAread_23_disablecopyonread_training_6_adam_gru_3_gru_cell_kernel_v"/device:CPU:0*
_output_shapes
 ├
Read_23/ReadVariableOpReadVariableOpAread_23_disablecopyonread_training_6_adam_gru_3_gru_cell_kernel_v^Read_23/DisableCopyOnRead"/device:CPU:0*
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

:`а
Read_24/DisableCopyOnReadDisableCopyOnReadKread_24_disablecopyonread_training_6_adam_gru_3_gru_cell_recurrent_kernel_v"/device:CPU:0*
_output_shapes
 ═
Read_24/ReadVariableOpReadVariableOpKread_24_disablecopyonread_training_6_adam_gru_3_gru_cell_recurrent_kernel_v^Read_24/DisableCopyOnRead"/device:CPU:0*
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

: `Ф
Read_25/DisableCopyOnReadDisableCopyOnRead?read_25_disablecopyonread_training_6_adam_gru_3_gru_cell_bias_v"/device:CPU:0*
_output_shapes
 ┴
Read_25/ReadVariableOpReadVariableOp?read_25_disablecopyonread_training_6_adam_gru_3_gru_cell_bias_v^Read_25/DisableCopyOnRead"/device:CPU:0*
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

:`г
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*╠
value┬B┐B6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/1/true_positives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/1/true_negatives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/1/false_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/1/false_negatives/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHг
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*I
value@B>B B B B B B B B B B B B B B B B B B B B B B B B B B B й
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Identity_1:output:0Identity_3:output:0Identity_5:output:0Identity_7:output:0Identity_9:output:0Identity_11:output:0Identity_13:output:0Identity_15:output:0Identity_17:output:0Identity_19:output:0Identity_21:output:0Identity_23:output:0Identity_25:output:0Identity_27:output:0Identity_29:output:0Identity_31:output:0Identity_33:output:0Identity_35:output:0Identity_37:output:0Identity_39:output:0Identity_41:output:0Identity_43:output:0Identity_45:output:0Identity_47:output:0Identity_49:output:0Identity_51:output:0savev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *)
dtypes
2	Р
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:│
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
: Б
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

_user_specified_nameConst:EA
?
_user_specified_name'%training_6/Adam/gru_3/gru_cell/bias/v:QM
K
_user_specified_name31training_6/Adam/gru_3/gru_cell/recurrent_kernel/v:GC
A
_user_specified_name)'training_6/Adam/gru_3/gru_cell/kernel/v:?;
9
_user_specified_name!training_6/Adam/dense_19/bias/v:A=
;
_user_specified_name#!training_6/Adam/dense_19/kernel/v:EA
?
_user_specified_name'%training_6/Adam/gru_3/gru_cell/bias/m:QM
K
_user_specified_name31training_6/Adam/gru_3/gru_cell/recurrent_kernel/m:GC
A
_user_specified_name)'training_6/Adam/gru_3/gru_cell/kernel/m:?;
9
_user_specified_name!training_6/Adam/dense_19/bias/m:A=
;
_user_specified_name#!training_6/Adam/dense_19/kernel/m:1-
+
_user_specified_namefalse_negatives_3:1-
+
_user_specified_namefalse_positives_3:0,
*
_user_specified_nametrue_negatives_3:0,
*
_user_specified_nametrue_positives_3:'#
!
_user_specified_name	count_3:'#
!
_user_specified_name	total_3:=
9
7
_user_specified_nametraining_6/Adam/learning_rate:5	1
/
_user_specified_nametraining_6/Adam/decay:62
0
_user_specified_nametraining_6/Adam/beta_2:62
0
_user_specified_nametraining_6/Adam/beta_1:40
.
_user_specified_nametraining_6/Adam/iter:3/
-
_user_specified_namegru_3/gru_cell/bias:?;
9
_user_specified_name!gru_3/gru_cell/recurrent_kernel:51
/
_user_specified_namegru_3/gru_cell/kernel:-)
'
_user_specified_namedense_19/bias:/+
)
_user_specified_namedense_19/kernel:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
▐
q
E__inference_multiply_3_layer_call_and_return_conditional_losses_21533
inputs_0
inputs_1
identityU
mulMulinputs_0inputs_1*
T0*,
_output_shapes
:         │
T
IdentityIdentitymul:z:0*
T0*,
_output_shapes
:         │
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:         │
:         │
:VR
,
_output_shapes
:         │

"
_user_specified_name
inputs_1:V R
,
_output_shapes
:         │

"
_user_specified_name
inputs_0
Х"
С
C__inference_model_22_layer_call_and_return_conditional_losses_21207
input_prior+
gru_3_gru_3_gru_cell_bias:`-
gru_3_gru_3_gru_cell_kernel:`7
%gru_3_gru_3_gru_cell_recurrent_kernel: `*
dense_19_dense_19_kernel: $
dense_19_dense_19_bias:
identityИв dense_19/StatefulPartitionedCallвgru_3/StatefulPartitionedCall╦
masking_prior/PartitionedCallPartitionedCallinput_prior*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         │
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_masking_prior_layer_call_and_return_conditional_losses_20992А
dot_6/PartitionedCallPartitionedCall&masking_prior/PartitionedCall:output:0&masking_prior/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:         ││* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *I
fDRB
@__inference_dot_6_layer_call_and_return_conditional_losses_21002▌
activation_3/PartitionedCallPartitionedCalldot_6/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:         ││* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_activation_3_layer_call_and_return_conditional_losses_21008■
dot_7/PartitionedCallPartitionedCall%activation_3/PartitionedCall:output:0&masking_prior/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         │
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *I
fDRB
@__inference_dot_7_layer_call_and_return_conditional_losses_21016Б
multiply_3/PartitionedCallPartitionedCalldot_7/PartitionedCall:output:0&masking_prior/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         │
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_multiply_3_layer_call_and_return_conditional_losses_21023М
concatenate_3/PartitionedCallPartitionedCall&masking_prior/PartitionedCall:output:0#multiply_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         │* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_concatenate_3_layer_call_and_return_conditional_losses_21031╚
gru_3/StatefulPartitionedCallStatefulPartitionedCall&concatenate_3/PartitionedCall:output:0gru_3_gru_3_gru_cell_biasgru_3_gru_3_gru_cell_kernel%gru_3_gru_3_gru_cell_recurrent_kernel*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:          *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *I
fDRB
@__inference_gru_3_layer_call_and_return_conditional_losses_21187Я
 dense_19/StatefulPartitionedCallStatefulPartitionedCall&gru_3/StatefulPartitionedCall:output:0dense_19_dense_19_kerneldense_19_dense_19_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_dense_19_layer_call_and_return_conditional_losses_21202x
IdentityIdentity)dense_19/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         e
NoOpNoOp!^dense_19/StatefulPartitionedCall^gru_3/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":         │
: : : : : 2D
 dense_19/StatefulPartitionedCall dense_19/StatefulPartitionedCall2>
gru_3/StatefulPartitionedCallgru_3/StatefulPartitionedCall:-)
'
_user_specified_namedense_19/bias:/+
)
_user_specified_namedense_19/kernel:?;
9
_user_specified_name!gru_3/gru_cell/recurrent_kernel:51
/
_user_specified_namegru_3/gru_cell/kernel:3/
-
_user_specified_namegru_3/gru_cell/bias:Y U
,
_output_shapes
:         │

%
_user_specified_nameinput_prior
┘9
║
while_body_22108
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
strided_slice_1_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0?
-gru_cell_readvariableop_gru_3_gru_cell_bias_0:`H
6gru_cell_matmul_readvariableop_gru_3_gru_cell_kernel_0:`T
Bgru_cell_matmul_1_readvariableop_gru_3_gru_cell_recurrent_kernel_0: `
identity

identity_1

identity_2

identity_3

identity_4
strided_slice_1O
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor=
+gru_cell_readvariableop_gru_3_gru_cell_bias:`F
4gru_cell_matmul_readvariableop_gru_3_gru_cell_kernel:`R
@gru_cell_matmul_1_readvariableop_gru_3_gru_cell_recurrent_kernel: `Ивgru_cell/MatMul/ReadVariableOpв gru_cell/MatMul_1/ReadVariableOpвgru_cell/ReadVariableOpВ
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       О
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0Е
gru_cell/ReadVariableOpReadVariableOp-gru_cell_readvariableop_gru_3_gru_cell_bias_0*
_output_shapes

:`*
dtype0q
gru_cell/unstackUnpackgru_cell/ReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
numХ
gru_cell/MatMul/ReadVariableOpReadVariableOp6gru_cell_matmul_readvariableop_gru_3_gru_cell_kernel_0*
_output_shapes

:`*
dtype0Я
gru_cell/MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0&gru_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         `Г
gru_cell/BiasAddBiasAddgru_cell/MatMul:product:0gru_cell/unstack:output:0*
T0*'
_output_shapes
:         `c
gru_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         ╛
gru_cell/splitSplit!gru_cell/split/split_dim:output:0gru_cell/BiasAdd:output:0*
T0*M
_output_shapes;
9:          :          :          *
	num_splitг
 gru_cell/MatMul_1/ReadVariableOpReadVariableOpBgru_cell_matmul_1_readvariableop_gru_3_gru_cell_recurrent_kernel_0*
_output_shapes

: `*
dtype0Ж
gru_cell/MatMul_1MatMulplaceholder_2(gru_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         `З
gru_cell/BiasAdd_1BiasAddgru_cell/MatMul_1:product:0gru_cell/unstack:output:1*
T0*'
_output_shapes
:         `c
gru_cell/ConstConst*
_output_shapes
:*
dtype0*!
valueB"            e
gru_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         ъ
gru_cell/split_1SplitVgru_cell/BiasAdd_1:output:0gru_cell/Const:output:0#gru_cell/split_1/split_dim:output:0*

Tlen0*
T0*M
_output_shapes;
9:          :          :          *
	num_split{
gru_cell/addAddV2gru_cell/split:output:0gru_cell/split_1:output:0*
T0*'
_output_shapes
:          _
gru_cell/SigmoidSigmoidgru_cell/add:z:0*
T0*'
_output_shapes
:          }
gru_cell/add_1AddV2gru_cell/split:output:1gru_cell/split_1:output:1*
T0*'
_output_shapes
:          c
gru_cell/Sigmoid_1Sigmoidgru_cell/add_1:z:0*
T0*'
_output_shapes
:          x
gru_cell/mulMulgru_cell/Sigmoid_1:y:0gru_cell/split_1:output:2*
T0*'
_output_shapes
:          t
gru_cell/add_2AddV2gru_cell/split:output:2gru_cell/mul:z:0*
T0*'
_output_shapes
:          [
gru_cell/TanhTanhgru_cell/add_2:z:0*
T0*'
_output_shapes
:          l
gru_cell/mul_1Mulgru_cell/Sigmoid:y:0placeholder_2*
T0*'
_output_shapes
:          S
gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?t
gru_cell/subSubgru_cell/sub/x:output:0gru_cell/Sigmoid:y:0*
T0*'
_output_shapes
:          l
gru_cell/mul_2Mulgru_cell/sub:z:0gru_cell/Tanh:y:0*
T0*'
_output_shapes
:          q
gru_cell/add_3AddV2gru_cell/mul_1:z:0gru_cell/mul_2:z:0*
T0*'
_output_shapes
:          l
*TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : ╤
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_13TensorArrayV2Write/TensorListSetItem/index:output:0gru_cell/add_3:z:0*
_output_shapes
: *
element_dtype0:щш╥G
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
:          А
NoOpNoOp^gru_cell/MatMul/ReadVariableOp!^gru_cell/MatMul_1/ReadVariableOp^gru_cell/ReadVariableOp*
_output_shapes
 "Ж
@gru_cell_matmul_1_readvariableop_gru_3_gru_cell_recurrent_kernelBgru_cell_matmul_1_readvariableop_gru_3_gru_cell_recurrent_kernel_0"n
4gru_cell_matmul_readvariableop_gru_3_gru_cell_kernel6gru_cell_matmul_readvariableop_gru_3_gru_cell_kernel_0"\
+gru_cell_readvariableop_gru_3_gru_cell_bias-gru_cell_readvariableop_gru_3_gru_cell_bias_0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
identityIdentity:output:0"$
strided_slice_1strided_slice_1_0"Ь
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :          : : : : : 2@
gru_cell/MatMul/ReadVariableOpgru_cell/MatMul/ReadVariableOp2D
 gru_cell/MatMul_1/ReadVariableOp gru_cell/MatMul_1/ReadVariableOp22
gru_cell/ReadVariableOpgru_cell/ReadVariableOp:?	;
9
_user_specified_name!gru_3/gru_cell/recurrent_kernel:51
/
_user_specified_namegru_3/gru_cell/kernel:3/
-
_user_specified_namegru_3/gru_cell/bias:_[
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
:          :
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
а
l
@__inference_dot_6_layer_call_and_return_conditional_losses_21498
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
:         
│h
MatMulBatchMatMulV2inputs_0transpose:y:0*
T0*-
_output_shapes
:         ││R
ShapeShapeMatMul:output:0*
T0*
_output_shapes
::э╧]
IdentityIdentityMatMul:output:0*
T0*-
_output_shapes
:         ││"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:         │
:         │
:VR
,
_output_shapes
:         │

"
_user_specified_name
inputs_1:V R
,
_output_shapes
:         │

"
_user_specified_name
inputs_0
╒J
Р

model_22_gru_3_while_body_20553%
!model_22_gru_3_while_loop_counter+
'model_22_gru_3_while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3$
 model_22_gru_3_strided_slice_1_0`
\tensorarrayv2read_tensorlistgetitem_model_22_gru_3_tensorarrayunstack_tensorlistfromtensor_0d
`tensorarrayv2read_1_tensorlistgetitem_model_22_gru_3_tensorarrayunstack_1_tensorlistfromtensor_0?
-gru_cell_readvariableop_gru_3_gru_cell_bias_0:`H
6gru_cell_matmul_readvariableop_gru_3_gru_cell_kernel_0:`T
Bgru_cell_matmul_1_readvariableop_gru_3_gru_cell_recurrent_kernel_0: `
identity

identity_1

identity_2

identity_3

identity_4

identity_5"
model_22_gru_3_strided_slice_1^
Ztensorarrayv2read_tensorlistgetitem_model_22_gru_3_tensorarrayunstack_tensorlistfromtensorb
^tensorarrayv2read_1_tensorlistgetitem_model_22_gru_3_tensorarrayunstack_1_tensorlistfromtensor=
+gru_cell_readvariableop_gru_3_gru_cell_bias:`F
4gru_cell_matmul_readvariableop_gru_3_gru_cell_kernel:`R
@gru_cell_matmul_1_readvariableop_gru_3_gru_cell_recurrent_kernel: `Ивgru_cell/MatMul/ReadVariableOpв gru_cell/MatMul_1/ReadVariableOpвgru_cell/ReadVariableOpВ
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       Э
#TensorArrayV2Read/TensorListGetItemTensorListGetItem\tensorarrayv2read_tensorlistgetitem_model_22_gru_3_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0Д
3TensorArrayV2Read_1/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       е
%TensorArrayV2Read_1/TensorListGetItemTensorListGetItem`tensorarrayv2read_1_tensorlistgetitem_model_22_gru_3_tensorarrayunstack_1_tensorlistfromtensor_0placeholder<TensorArrayV2Read_1/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0
Е
gru_cell/ReadVariableOpReadVariableOp-gru_cell_readvariableop_gru_3_gru_cell_bias_0*
_output_shapes

:`*
dtype0q
gru_cell/unstackUnpackgru_cell/ReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
numХ
gru_cell/MatMul/ReadVariableOpReadVariableOp6gru_cell_matmul_readvariableop_gru_3_gru_cell_kernel_0*
_output_shapes

:`*
dtype0Я
gru_cell/MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0&gru_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         `Г
gru_cell/BiasAddBiasAddgru_cell/MatMul:product:0gru_cell/unstack:output:0*
T0*'
_output_shapes
:         `c
gru_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         ╛
gru_cell/splitSplit!gru_cell/split/split_dim:output:0gru_cell/BiasAdd:output:0*
T0*M
_output_shapes;
9:          :          :          *
	num_splitг
 gru_cell/MatMul_1/ReadVariableOpReadVariableOpBgru_cell_matmul_1_readvariableop_gru_3_gru_cell_recurrent_kernel_0*
_output_shapes

: `*
dtype0Ж
gru_cell/MatMul_1MatMulplaceholder_3(gru_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         `З
gru_cell/BiasAdd_1BiasAddgru_cell/MatMul_1:product:0gru_cell/unstack:output:1*
T0*'
_output_shapes
:         `c
gru_cell/ConstConst*
_output_shapes
:*
dtype0*!
valueB"            e
gru_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         ъ
gru_cell/split_1SplitVgru_cell/BiasAdd_1:output:0gru_cell/Const:output:0#gru_cell/split_1/split_dim:output:0*

Tlen0*
T0*M
_output_shapes;
9:          :          :          *
	num_split{
gru_cell/addAddV2gru_cell/split:output:0gru_cell/split_1:output:0*
T0*'
_output_shapes
:          _
gru_cell/SigmoidSigmoidgru_cell/add:z:0*
T0*'
_output_shapes
:          }
gru_cell/add_1AddV2gru_cell/split:output:1gru_cell/split_1:output:1*
T0*'
_output_shapes
:          c
gru_cell/Sigmoid_1Sigmoidgru_cell/add_1:z:0*
T0*'
_output_shapes
:          x
gru_cell/mulMulgru_cell/Sigmoid_1:y:0gru_cell/split_1:output:2*
T0*'
_output_shapes
:          t
gru_cell/add_2AddV2gru_cell/split:output:2gru_cell/mul:z:0*
T0*'
_output_shapes
:          [
gru_cell/TanhTanhgru_cell/add_2:z:0*
T0*'
_output_shapes
:          l
gru_cell/mul_1Mulgru_cell/Sigmoid:y:0placeholder_3*
T0*'
_output_shapes
:          S
gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?t
gru_cell/subSubgru_cell/sub/x:output:0gru_cell/Sigmoid:y:0*
T0*'
_output_shapes
:          l
gru_cell/mul_2Mulgru_cell/sub:z:0gru_cell/Tanh:y:0*
T0*'
_output_shapes
:          q
gru_cell/add_3AddV2gru_cell/mul_1:z:0gru_cell/mul_2:z:0*
T0*'
_output_shapes
:          _
Tile/multiplesConst*
_output_shapes
:*
dtype0*
valueB"      Е
TileTile,TensorArrayV2Read_1/TensorListGetItem:item:0Tile/multiples:output:0*
T0
*'
_output_shapes
:         x
SelectV2SelectV2Tile:output:0gru_cell/add_3:z:0placeholder_2*
T0*'
_output_shapes
:          a
Tile_1/multiplesConst*
_output_shapes
:*
dtype0*
valueB"      Й
Tile_1Tile,TensorArrayV2Read_1/TensorListGetItem:item:0Tile_1/multiples:output:0*
T0
*'
_output_shapes
:         |

SelectV2_1SelectV2Tile_1:output:0gru_cell/add_3:z:0placeholder_3*
T0*'
_output_shapes
:          l
*TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : ╨
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_13TensorArrayV2Write/TensorListSetItem/index:output:0SelectV2:output:0*
_output_shapes
: *
element_dtype0:щш╥G
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
add_1AddV2!model_22_gru_3_while_loop_counteradd_1/y:output:0*
T0*
_output_shapes
: G
IdentityIdentity	add_1:z:0^NoOp*
T0*
_output_shapes
: g

Identity_1Identity'model_22_gru_3_while_maximum_iterations^NoOp*
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
:          d

Identity_5IdentitySelectV2_1:output:0^NoOp*
T0*'
_output_shapes
:          А
NoOpNoOp^gru_cell/MatMul/ReadVariableOp!^gru_cell/MatMul_1/ReadVariableOp^gru_cell/ReadVariableOp*
_output_shapes
 "Ж
@gru_cell_matmul_1_readvariableop_gru_3_gru_cell_recurrent_kernelBgru_cell_matmul_1_readvariableop_gru_3_gru_cell_recurrent_kernel_0"n
4gru_cell_matmul_readvariableop_gru_3_gru_cell_kernel6gru_cell_matmul_readvariableop_gru_3_gru_cell_kernel_0"\
+gru_cell_readvariableop_gru_3_gru_cell_bias-gru_cell_readvariableop_gru_3_gru_cell_bias_0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"
identityIdentity:output:0"B
model_22_gru_3_strided_slice_1 model_22_gru_3_strided_slice_1_0"┬
^tensorarrayv2read_1_tensorlistgetitem_model_22_gru_3_tensorarrayunstack_1_tensorlistfromtensor`tensorarrayv2read_1_tensorlistgetitem_model_22_gru_3_tensorarrayunstack_1_tensorlistfromtensor_0"║
Ztensorarrayv2read_tensorlistgetitem_model_22_gru_3_tensorarrayunstack_tensorlistfromtensor\tensorarrayv2read_tensorlistgetitem_model_22_gru_3_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :          :          : : : : : : 2@
gru_cell/MatMul/ReadVariableOpgru_cell/MatMul/ReadVariableOp2D
 gru_cell/MatMul_1/ReadVariableOp gru_cell/MatMul_1/ReadVariableOp22
gru_cell/ReadVariableOpgru_cell/ReadVariableOp:?;
9
_user_specified_name!gru_3/gru_cell/recurrent_kernel:5
1
/
_user_specified_namegru_3/gru_cell/kernel:3	/
-
_user_specified_namegru_3/gru_cell/bias:pl

_output_shapes
: 
R
_user_specified_name:8model_22/gru_3/TensorArrayUnstack_1/TensorListFromTensor:nj

_output_shapes
: 
P
_user_specified_name86model_22/gru_3/TensorArrayUnstack/TensorListFromTensor:VR

_output_shapes
: 
8
_user_specified_name model_22/gru_3/strided_slice_1:-)
'
_output_shapes
:          :-)
'
_output_shapes
:          :
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
_user_specified_name)'model_22/gru_3/while/maximum_iterations:Y U

_output_shapes
: 
;
_user_specified_name#!model_22/gru_3/while/loop_counter
Ў
у
while_cond_21279
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
less_strided_slice_1-
)while_cond_21279___redundant_placeholder0-
)while_cond_21279___redundant_placeholder1-
)while_cond_21279___redundant_placeholder2-
)while_cond_21279___redundant_placeholder3
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
-: : : : :          : :::::
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
:          :
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
╝
j
@__inference_dot_7_layer_call_and_return_conditional_losses_21016

inputs
inputs_1
identity`
MatMulBatchMatMulV2inputsinputs_1*
T0*,
_output_shapes
:         │
R
ShapeShapeMatMul:output:0*
T0*
_output_shapes
::э╧\
IdentityIdentityMatMul:output:0*
T0*,
_output_shapes
:         │
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:         ││:         │
:TP
,
_output_shapes
:         │

 
_user_specified_nameinputs:U Q
-
_output_shapes
:         ││
 
_user_specified_nameinputs
┘	
с
%__inference_gru_3_layer_call_fn_21578

inputs%
gru_3_gru_cell_bias:`'
gru_3_gru_cell_kernel:`1
gru_3_gru_cell_recurrent_kernel: `
identityИвStatefulPartitionedCallР
StatefulPartitionedCallStatefulPartitionedCallinputsgru_3_gru_cell_biasgru_3_gru_cell_kernelgru_3_gru_cell_recurrent_kernel*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:          *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *I
fDRB
@__inference_gru_3_layer_call_and_return_conditional_losses_21370o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:          <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:         │: : : 22
StatefulPartitionedCallStatefulPartitionedCall:?;
9
_user_specified_name!gru_3/gru_cell/recurrent_kernel:51
/
_user_specified_namegru_3/gru_cell/kernel:3/
-
_user_specified_namegru_3/gru_cell/bias:T P
,
_output_shapes
:         │
 
_user_specified_nameinputs
┬
V
*__inference_multiply_3_layer_call_fn_21527
inputs_0
inputs_1
identity┬
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         │
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_multiply_3_layer_call_and_return_conditional_losses_21023e
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:         │
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:         │
:         │
:VR
,
_output_shapes
:         │

"
_user_specified_name
inputs_1:V R
,
_output_shapes
:         │

"
_user_specified_name
inputs_0
ю6
и
@__inference_gru_3_layer_call_and_return_conditional_losses_20803

inputs.
gru_cell_gru_3_gru_cell_bias:`0
gru_cell_gru_3_gru_cell_kernel:`:
(gru_cell_gru_3_gru_cell_recurrent_kernel: `
identityИв gru_cell/StatefulPartitionedCallвwhileI
ShapeShapeinputs*
T0*
_output_shapes
::э╧]
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
valueB:╤
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
:          c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :                  R
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
::э╧_
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
valueB:█
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
         ┤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥Ж
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       р
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥_
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
:         *
shrink_axis_maskю
 gru_cell/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0gru_cell_gru_3_gru_cell_biasgru_cell_gru_3_gru_cell_kernel(gru_cell_gru_3_gru_cell_recurrent_kernel*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:          :          *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_gru_cell_layer_call_and_return_conditional_losses_20730n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :┼
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥F
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
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : И
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0gru_cell_gru_3_gru_cell_biasgru_cell_gru_3_gru_cell_kernel(gru_cell_gru_3_gru_cell_recurrent_kernel*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :          : : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_20741*
condR
while_cond_20740*8
output_shapes'
%: : : : :          : : : : : *
parallel_iterations Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        ╓
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:          *
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:З
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:          *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ц
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:          [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:          M
NoOpNoOp!^gru_cell/StatefulPartitionedCall^while*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  : : : 2D
 gru_cell/StatefulPartitionedCall gru_cell/StatefulPartitionedCall2
whilewhile:?;
9
_user_specified_name!gru_3/gru_cell/recurrent_kernel:51
/
_user_specified_namegru_3/gru_cell/kernel:3/
-
_user_specified_namegru_3/gru_cell/bias:\ X
4
_output_shapes"
 :                  
 
_user_specified_nameinputs
┘9
║
while_body_21280
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
strided_slice_1_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0?
-gru_cell_readvariableop_gru_3_gru_cell_bias_0:`H
6gru_cell_matmul_readvariableop_gru_3_gru_cell_kernel_0:`T
Bgru_cell_matmul_1_readvariableop_gru_3_gru_cell_recurrent_kernel_0: `
identity

identity_1

identity_2

identity_3

identity_4
strided_slice_1O
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor=
+gru_cell_readvariableop_gru_3_gru_cell_bias:`F
4gru_cell_matmul_readvariableop_gru_3_gru_cell_kernel:`R
@gru_cell_matmul_1_readvariableop_gru_3_gru_cell_recurrent_kernel: `Ивgru_cell/MatMul/ReadVariableOpв gru_cell/MatMul_1/ReadVariableOpвgru_cell/ReadVariableOpВ
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       О
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0Е
gru_cell/ReadVariableOpReadVariableOp-gru_cell_readvariableop_gru_3_gru_cell_bias_0*
_output_shapes

:`*
dtype0q
gru_cell/unstackUnpackgru_cell/ReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
numХ
gru_cell/MatMul/ReadVariableOpReadVariableOp6gru_cell_matmul_readvariableop_gru_3_gru_cell_kernel_0*
_output_shapes

:`*
dtype0Я
gru_cell/MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0&gru_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         `Г
gru_cell/BiasAddBiasAddgru_cell/MatMul:product:0gru_cell/unstack:output:0*
T0*'
_output_shapes
:         `c
gru_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         ╛
gru_cell/splitSplit!gru_cell/split/split_dim:output:0gru_cell/BiasAdd:output:0*
T0*M
_output_shapes;
9:          :          :          *
	num_splitг
 gru_cell/MatMul_1/ReadVariableOpReadVariableOpBgru_cell_matmul_1_readvariableop_gru_3_gru_cell_recurrent_kernel_0*
_output_shapes

: `*
dtype0Ж
gru_cell/MatMul_1MatMulplaceholder_2(gru_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         `З
gru_cell/BiasAdd_1BiasAddgru_cell/MatMul_1:product:0gru_cell/unstack:output:1*
T0*'
_output_shapes
:         `c
gru_cell/ConstConst*
_output_shapes
:*
dtype0*!
valueB"            e
gru_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         ъ
gru_cell/split_1SplitVgru_cell/BiasAdd_1:output:0gru_cell/Const:output:0#gru_cell/split_1/split_dim:output:0*

Tlen0*
T0*M
_output_shapes;
9:          :          :          *
	num_split{
gru_cell/addAddV2gru_cell/split:output:0gru_cell/split_1:output:0*
T0*'
_output_shapes
:          _
gru_cell/SigmoidSigmoidgru_cell/add:z:0*
T0*'
_output_shapes
:          }
gru_cell/add_1AddV2gru_cell/split:output:1gru_cell/split_1:output:1*
T0*'
_output_shapes
:          c
gru_cell/Sigmoid_1Sigmoidgru_cell/add_1:z:0*
T0*'
_output_shapes
:          x
gru_cell/mulMulgru_cell/Sigmoid_1:y:0gru_cell/split_1:output:2*
T0*'
_output_shapes
:          t
gru_cell/add_2AddV2gru_cell/split:output:2gru_cell/mul:z:0*
T0*'
_output_shapes
:          [
gru_cell/TanhTanhgru_cell/add_2:z:0*
T0*'
_output_shapes
:          l
gru_cell/mul_1Mulgru_cell/Sigmoid:y:0placeholder_2*
T0*'
_output_shapes
:          S
gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?t
gru_cell/subSubgru_cell/sub/x:output:0gru_cell/Sigmoid:y:0*
T0*'
_output_shapes
:          l
gru_cell/mul_2Mulgru_cell/sub:z:0gru_cell/Tanh:y:0*
T0*'
_output_shapes
:          q
gru_cell/add_3AddV2gru_cell/mul_1:z:0gru_cell/mul_2:z:0*
T0*'
_output_shapes
:          l
*TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : ╤
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_13TensorArrayV2Write/TensorListSetItem/index:output:0gru_cell/add_3:z:0*
_output_shapes
: *
element_dtype0:щш╥G
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
:          А
NoOpNoOp^gru_cell/MatMul/ReadVariableOp!^gru_cell/MatMul_1/ReadVariableOp^gru_cell/ReadVariableOp*
_output_shapes
 "Ж
@gru_cell_matmul_1_readvariableop_gru_3_gru_cell_recurrent_kernelBgru_cell_matmul_1_readvariableop_gru_3_gru_cell_recurrent_kernel_0"n
4gru_cell_matmul_readvariableop_gru_3_gru_cell_kernel6gru_cell_matmul_readvariableop_gru_3_gru_cell_kernel_0"\
+gru_cell_readvariableop_gru_3_gru_cell_bias-gru_cell_readvariableop_gru_3_gru_cell_bias_0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
identityIdentity:output:0"$
strided_slice_1strided_slice_1_0"Ь
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :          : : : : : 2@
gru_cell/MatMul/ReadVariableOpgru_cell/MatMul/ReadVariableOp2D
 gru_cell/MatMul_1/ReadVariableOp gru_cell/MatMul_1/ReadVariableOp22
gru_cell/ReadVariableOpgru_cell/ReadVariableOp:?	;
9
_user_specified_name!gru_3/gru_cell/recurrent_kernel:51
/
_user_specified_namegru_3/gru_cell/kernel:3/
-
_user_specified_namegru_3/gru_cell/bias:_[
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
:          :
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
▓N
а
@__inference_gru_3_layer_call_and_return_conditional_losses_22043

inputs=
+gru_cell_readvariableop_gru_3_gru_cell_bias:`F
4gru_cell_matmul_readvariableop_gru_3_gru_cell_kernel:`R
@gru_cell_matmul_1_readvariableop_gru_3_gru_cell_recurrent_kernel: `
identityИвgru_cell/MatMul/ReadVariableOpв gru_cell/MatMul_1/ReadVariableOpвgru_cell/ReadVariableOpвwhileI
ShapeShapeinputs*
T0*
_output_shapes
::э╧]
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
valueB:╤
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
:          c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          n
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:│         R
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
::э╧_
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
valueB:█
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
         ┤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥Ж
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       р
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥_
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
:         *
shrink_axis_maskГ
gru_cell/ReadVariableOpReadVariableOp+gru_cell_readvariableop_gru_3_gru_cell_bias*
_output_shapes

:`*
dtype0q
gru_cell/unstackUnpackgru_cell/ReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
numУ
gru_cell/MatMul/ReadVariableOpReadVariableOp4gru_cell_matmul_readvariableop_gru_3_gru_cell_kernel*
_output_shapes

:`*
dtype0Н
gru_cell/MatMulMatMulstrided_slice_2:output:0&gru_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         `Г
gru_cell/BiasAddBiasAddgru_cell/MatMul:product:0gru_cell/unstack:output:0*
T0*'
_output_shapes
:         `c
gru_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         ╛
gru_cell/splitSplit!gru_cell/split/split_dim:output:0gru_cell/BiasAdd:output:0*
T0*M
_output_shapes;
9:          :          :          *
	num_splitб
 gru_cell/MatMul_1/ReadVariableOpReadVariableOp@gru_cell_matmul_1_readvariableop_gru_3_gru_cell_recurrent_kernel*
_output_shapes

: `*
dtype0З
gru_cell/MatMul_1MatMulzeros:output:0(gru_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         `З
gru_cell/BiasAdd_1BiasAddgru_cell/MatMul_1:product:0gru_cell/unstack:output:1*
T0*'
_output_shapes
:         `c
gru_cell/ConstConst*
_output_shapes
:*
dtype0*!
valueB"            e
gru_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         ъ
gru_cell/split_1SplitVgru_cell/BiasAdd_1:output:0gru_cell/Const:output:0#gru_cell/split_1/split_dim:output:0*

Tlen0*
T0*M
_output_shapes;
9:          :          :          *
	num_split{
gru_cell/addAddV2gru_cell/split:output:0gru_cell/split_1:output:0*
T0*'
_output_shapes
:          _
gru_cell/SigmoidSigmoidgru_cell/add:z:0*
T0*'
_output_shapes
:          }
gru_cell/add_1AddV2gru_cell/split:output:1gru_cell/split_1:output:1*
T0*'
_output_shapes
:          c
gru_cell/Sigmoid_1Sigmoidgru_cell/add_1:z:0*
T0*'
_output_shapes
:          x
gru_cell/mulMulgru_cell/Sigmoid_1:y:0gru_cell/split_1:output:2*
T0*'
_output_shapes
:          t
gru_cell/add_2AddV2gru_cell/split:output:2gru_cell/mul:z:0*
T0*'
_output_shapes
:          [
gru_cell/TanhTanhgru_cell/add_2:z:0*
T0*'
_output_shapes
:          m
gru_cell/mul_1Mulgru_cell/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:          S
gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?t
gru_cell/subSubgru_cell/sub/x:output:0gru_cell/Sigmoid:y:0*
T0*'
_output_shapes
:          l
gru_cell/mul_2Mulgru_cell/sub:z:0gru_cell/Tanh:y:0*
T0*'
_output_shapes
:          q
gru_cell/add_3AddV2gru_cell/mul_1:z:0gru_cell/mul_2:z:0*
T0*'
_output_shapes
:          n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :┼
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥F
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
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ┼
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+gru_cell_readvariableop_gru_3_gru_cell_bias4gru_cell_matmul_readvariableop_gru_3_gru_cell_kernel@gru_cell_matmul_1_readvariableop_gru_3_gru_cell_recurrent_kernel*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :          : : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_21953*
condR
while_cond_21952*8
output_shapes'
%: : : : :          : : : : : *
parallel_iterations Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        ╓
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:          *
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:З
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:          *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ц
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:          [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:          И
NoOpNoOp^gru_cell/MatMul/ReadVariableOp!^gru_cell/MatMul_1/ReadVariableOp^gru_cell/ReadVariableOp^while*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:         │: : : 2@
gru_cell/MatMul/ReadVariableOpgru_cell/MatMul/ReadVariableOp2D
 gru_cell/MatMul_1/ReadVariableOp gru_cell/MatMul_1/ReadVariableOp22
gru_cell/ReadVariableOpgru_cell/ReadVariableOp2
whilewhile:?;
9
_user_specified_name!gru_3/gru_cell/recurrent_kernel:51
/
_user_specified_namegru_3/gru_cell/kernel:3/
-
_user_specified_namegru_3/gru_cell/bias:T P
,
_output_shapes
:         │
 
_user_specified_nameinputs
Ў
у
while_cond_20740
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
less_strided_slice_1-
)while_cond_20740___redundant_placeholder0-
)while_cond_20740___redundant_placeholder1-
)while_cond_20740___redundant_placeholder2-
)while_cond_20740___redundant_placeholder3
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
-: : : : :          : :::::
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
:          :
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
ю6
и
@__inference_gru_3_layer_call_and_return_conditional_losses_20941

inputs.
gru_cell_gru_3_gru_cell_bias:`0
gru_cell_gru_3_gru_cell_kernel:`:
(gru_cell_gru_3_gru_cell_recurrent_kernel: `
identityИв gru_cell/StatefulPartitionedCallвwhileI
ShapeShapeinputs*
T0*
_output_shapes
::э╧]
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
valueB:╤
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
:          c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :                  R
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
::э╧_
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
valueB:█
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
         ┤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥Ж
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       р
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥_
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
:         *
shrink_axis_maskю
 gru_cell/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0gru_cell_gru_3_gru_cell_biasgru_cell_gru_3_gru_cell_kernel(gru_cell_gru_3_gru_cell_recurrent_kernel*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:          :          *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_gru_cell_layer_call_and_return_conditional_losses_20868n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :┼
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥F
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
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : И
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0gru_cell_gru_3_gru_cell_biasgru_cell_gru_3_gru_cell_kernel(gru_cell_gru_3_gru_cell_recurrent_kernel*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :          : : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_20879*
condR
while_cond_20878*8
output_shapes'
%: : : : :          : : : : : *
parallel_iterations Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        ╓
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:          *
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:З
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:          *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ц
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:          [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:          M
NoOpNoOp!^gru_cell/StatefulPartitionedCall^while*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  : : : 2D
 gru_cell/StatefulPartitionedCall gru_cell/StatefulPartitionedCall2
whilewhile:?;
9
_user_specified_name!gru_3/gru_cell/recurrent_kernel:51
/
_user_specified_namegru_3/gru_cell/kernel:3/
-
_user_specified_namegru_3/gru_cell/bias:\ X
4
_output_shapes"
 :                  
 
_user_specified_nameinputs
╫
r
H__inference_concatenate_3_layer_call_and_return_conditional_losses_21031

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
:         │\
IdentityIdentityconcat:output:0*
T0*,
_output_shapes
:         │"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:         │
:         │
:TP
,
_output_shapes
:         │

 
_user_specified_nameinputs:T P
,
_output_shapes
:         │

 
_user_specified_nameinputs
▓N
а
@__inference_gru_3_layer_call_and_return_conditional_losses_21370

inputs=
+gru_cell_readvariableop_gru_3_gru_cell_bias:`F
4gru_cell_matmul_readvariableop_gru_3_gru_cell_kernel:`R
@gru_cell_matmul_1_readvariableop_gru_3_gru_cell_recurrent_kernel: `
identityИвgru_cell/MatMul/ReadVariableOpв gru_cell/MatMul_1/ReadVariableOpвgru_cell/ReadVariableOpвwhileI
ShapeShapeinputs*
T0*
_output_shapes
::э╧]
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
valueB:╤
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
:          c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          n
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:│         R
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
::э╧_
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
valueB:█
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
         ┤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥Ж
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       р
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥_
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
:         *
shrink_axis_maskГ
gru_cell/ReadVariableOpReadVariableOp+gru_cell_readvariableop_gru_3_gru_cell_bias*
_output_shapes

:`*
dtype0q
gru_cell/unstackUnpackgru_cell/ReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
numУ
gru_cell/MatMul/ReadVariableOpReadVariableOp4gru_cell_matmul_readvariableop_gru_3_gru_cell_kernel*
_output_shapes

:`*
dtype0Н
gru_cell/MatMulMatMulstrided_slice_2:output:0&gru_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         `Г
gru_cell/BiasAddBiasAddgru_cell/MatMul:product:0gru_cell/unstack:output:0*
T0*'
_output_shapes
:         `c
gru_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         ╛
gru_cell/splitSplit!gru_cell/split/split_dim:output:0gru_cell/BiasAdd:output:0*
T0*M
_output_shapes;
9:          :          :          *
	num_splitб
 gru_cell/MatMul_1/ReadVariableOpReadVariableOp@gru_cell_matmul_1_readvariableop_gru_3_gru_cell_recurrent_kernel*
_output_shapes

: `*
dtype0З
gru_cell/MatMul_1MatMulzeros:output:0(gru_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         `З
gru_cell/BiasAdd_1BiasAddgru_cell/MatMul_1:product:0gru_cell/unstack:output:1*
T0*'
_output_shapes
:         `c
gru_cell/ConstConst*
_output_shapes
:*
dtype0*!
valueB"            e
gru_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         ъ
gru_cell/split_1SplitVgru_cell/BiasAdd_1:output:0gru_cell/Const:output:0#gru_cell/split_1/split_dim:output:0*

Tlen0*
T0*M
_output_shapes;
9:          :          :          *
	num_split{
gru_cell/addAddV2gru_cell/split:output:0gru_cell/split_1:output:0*
T0*'
_output_shapes
:          _
gru_cell/SigmoidSigmoidgru_cell/add:z:0*
T0*'
_output_shapes
:          }
gru_cell/add_1AddV2gru_cell/split:output:1gru_cell/split_1:output:1*
T0*'
_output_shapes
:          c
gru_cell/Sigmoid_1Sigmoidgru_cell/add_1:z:0*
T0*'
_output_shapes
:          x
gru_cell/mulMulgru_cell/Sigmoid_1:y:0gru_cell/split_1:output:2*
T0*'
_output_shapes
:          t
gru_cell/add_2AddV2gru_cell/split:output:2gru_cell/mul:z:0*
T0*'
_output_shapes
:          [
gru_cell/TanhTanhgru_cell/add_2:z:0*
T0*'
_output_shapes
:          m
gru_cell/mul_1Mulgru_cell/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:          S
gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?t
gru_cell/subSubgru_cell/sub/x:output:0gru_cell/Sigmoid:y:0*
T0*'
_output_shapes
:          l
gru_cell/mul_2Mulgru_cell/sub:z:0gru_cell/Tanh:y:0*
T0*'
_output_shapes
:          q
gru_cell/add_3AddV2gru_cell/mul_1:z:0gru_cell/mul_2:z:0*
T0*'
_output_shapes
:          n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :┼
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥F
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
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ┼
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+gru_cell_readvariableop_gru_3_gru_cell_bias4gru_cell_matmul_readvariableop_gru_3_gru_cell_kernel@gru_cell_matmul_1_readvariableop_gru_3_gru_cell_recurrent_kernel*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :          : : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_21280*
condR
while_cond_21279*8
output_shapes'
%: : : : :          : : : : : *
parallel_iterations Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        ╓
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:          *
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:З
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:          *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ц
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:          [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:          И
NoOpNoOp^gru_cell/MatMul/ReadVariableOp!^gru_cell/MatMul_1/ReadVariableOp^gru_cell/ReadVariableOp^while*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:         │: : : 2@
gru_cell/MatMul/ReadVariableOpgru_cell/MatMul/ReadVariableOp2D
 gru_cell/MatMul_1/ReadVariableOp gru_cell/MatMul_1/ReadVariableOp22
gru_cell/ReadVariableOpgru_cell/ReadVariableOp2
whilewhile:?;
9
_user_specified_name!gru_3/gru_cell/recurrent_kernel:51
/
_user_specified_namegru_3/gru_cell/kernel:3/
-
_user_specified_namegru_3/gru_cell/bias:T P
,
_output_shapes
:         │
 
_user_specified_nameinputs
Ў
у
while_cond_21642
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
less_strided_slice_1-
)while_cond_21642___redundant_placeholder0-
)while_cond_21642___redundant_placeholder1-
)while_cond_21642___redundant_placeholder2-
)while_cond_21642___redundant_placeholder3
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
-: : : : :          : :::::
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
:          :
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
▒!
Е
while_body_20741
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
strided_slice_1_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_00
gru_cell_gru_3_gru_cell_bias_0:`2
 gru_cell_gru_3_gru_cell_kernel_0:`<
*gru_cell_gru_3_gru_cell_recurrent_kernel_0: `
identity

identity_1

identity_2

identity_3

identity_4
strided_slice_1O
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor.
gru_cell_gru_3_gru_cell_bias:`0
gru_cell_gru_3_gru_cell_kernel:`:
(gru_cell_gru_3_gru_cell_recurrent_kernel: `Ив gru_cell/StatefulPartitionedCallВ
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       О
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0Е
 gru_cell/StatefulPartitionedCallStatefulPartitionedCall*TensorArrayV2Read/TensorListGetItem:item:0placeholder_2gru_cell_gru_3_gru_cell_bias_0 gru_cell_gru_3_gru_cell_kernel_0*gru_cell_gru_3_gru_cell_recurrent_kernel_0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:          :          *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_gru_cell_layer_call_and_return_conditional_losses_20730l
*TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : ш
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_13TensorArrayV2Write/TensorListSetItem/index:output:0)gru_cell/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:щш╥G
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
:          E
NoOpNoOp!^gru_cell/StatefulPartitionedCall*
_output_shapes
 ">
gru_cell_gru_3_gru_cell_biasgru_cell_gru_3_gru_cell_bias_0"B
gru_cell_gru_3_gru_cell_kernel gru_cell_gru_3_gru_cell_kernel_0"V
(gru_cell_gru_3_gru_cell_recurrent_kernel*gru_cell_gru_3_gru_cell_recurrent_kernel_0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
identityIdentity:output:0"$
strided_slice_1strided_slice_1_0"Ь
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :          : : : : : 2D
 gru_cell/StatefulPartitionedCall gru_cell/StatefulPartitionedCall:?	;
9
_user_specified_name!gru_3/gru_cell/recurrent_kernel:51
/
_user_specified_namegru_3/gru_cell/kernel:3/
-
_user_specified_namegru_3/gru_cell/bias:_[
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
:          :
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
Ў
у
while_cond_22107
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
less_strided_slice_1-
)while_cond_22107___redundant_placeholder0-
)while_cond_22107___redundant_placeholder1-
)while_cond_22107___redundant_placeholder2-
)while_cond_22107___redundant_placeholder3
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
-: : : : :          : :::::
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
:          :
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
║
Q
%__inference_dot_7_layer_call_fn_21514
inputs_0
inputs_1
identity╜
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         │
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *I
fDRB
@__inference_dot_7_layer_call_and_return_conditional_losses_21016e
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:         │
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:         ││:         │
:VR
,
_output_shapes
:         │

"
_user_specified_name
inputs_1:W S
-
_output_shapes
:         ││
"
_user_specified_name
inputs_0
є

А
C__inference_dense_19_layer_call_and_return_conditional_losses_22216

inputs7
%matmul_readvariableop_dense_19_kernel: 2
$biasadd_readvariableop_dense_19_bias:
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOp{
MatMul/ReadVariableOpReadVariableOp%matmul_readvariableop_dense_19_kernel*
_output_shapes

: *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         w
BiasAdd/ReadVariableOpReadVariableOp$biasadd_readvariableop_dense_19_bias*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:         `
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:         S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:          : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:-)
'
_user_specified_namedense_19/bias:/+
)
_user_specified_namedense_19/kernel:O K
'
_output_shapes
:          
 
_user_specified_nameinputs
О
д
#__inference_signature_wrapper_21467
input_prior%
gru_3_gru_cell_bias:`'
gru_3_gru_cell_kernel:`1
gru_3_gru_cell_recurrent_kernel: `!
dense_19_kernel: 
dense_19_bias:
identityИвStatefulPartitionedCallЩ
StatefulPartitionedCallStatefulPartitionedCallinput_priorgru_3_gru_cell_biasgru_3_gru_cell_kernelgru_3_gru_cell_recurrent_kerneldense_19_kerneldense_19_bias*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8В *)
f$R"
 __inference__wrapped_model_20665o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":         │
: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:-)
'
_user_specified_namedense_19/bias:/+
)
_user_specified_namedense_19/kernel:?;
9
_user_specified_name!gru_3/gru_cell/recurrent_kernel:51
/
_user_specified_namegru_3/gru_cell/kernel:3/
-
_user_specified_namegru_3/gru_cell/bias:Y U
,
_output_shapes
:         │

%
_user_specified_nameinput_prior
▓N
а
@__inference_gru_3_layer_call_and_return_conditional_losses_22198

inputs=
+gru_cell_readvariableop_gru_3_gru_cell_bias:`F
4gru_cell_matmul_readvariableop_gru_3_gru_cell_kernel:`R
@gru_cell_matmul_1_readvariableop_gru_3_gru_cell_recurrent_kernel: `
identityИвgru_cell/MatMul/ReadVariableOpв gru_cell/MatMul_1/ReadVariableOpвgru_cell/ReadVariableOpвwhileI
ShapeShapeinputs*
T0*
_output_shapes
::э╧]
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
valueB:╤
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
:          c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          n
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:│         R
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
::э╧_
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
valueB:█
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
         ┤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥Ж
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       р
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥_
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
:         *
shrink_axis_maskГ
gru_cell/ReadVariableOpReadVariableOp+gru_cell_readvariableop_gru_3_gru_cell_bias*
_output_shapes

:`*
dtype0q
gru_cell/unstackUnpackgru_cell/ReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
numУ
gru_cell/MatMul/ReadVariableOpReadVariableOp4gru_cell_matmul_readvariableop_gru_3_gru_cell_kernel*
_output_shapes

:`*
dtype0Н
gru_cell/MatMulMatMulstrided_slice_2:output:0&gru_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         `Г
gru_cell/BiasAddBiasAddgru_cell/MatMul:product:0gru_cell/unstack:output:0*
T0*'
_output_shapes
:         `c
gru_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         ╛
gru_cell/splitSplit!gru_cell/split/split_dim:output:0gru_cell/BiasAdd:output:0*
T0*M
_output_shapes;
9:          :          :          *
	num_splitб
 gru_cell/MatMul_1/ReadVariableOpReadVariableOp@gru_cell_matmul_1_readvariableop_gru_3_gru_cell_recurrent_kernel*
_output_shapes

: `*
dtype0З
gru_cell/MatMul_1MatMulzeros:output:0(gru_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         `З
gru_cell/BiasAdd_1BiasAddgru_cell/MatMul_1:product:0gru_cell/unstack:output:1*
T0*'
_output_shapes
:         `c
gru_cell/ConstConst*
_output_shapes
:*
dtype0*!
valueB"            e
gru_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         ъ
gru_cell/split_1SplitVgru_cell/BiasAdd_1:output:0gru_cell/Const:output:0#gru_cell/split_1/split_dim:output:0*

Tlen0*
T0*M
_output_shapes;
9:          :          :          *
	num_split{
gru_cell/addAddV2gru_cell/split:output:0gru_cell/split_1:output:0*
T0*'
_output_shapes
:          _
gru_cell/SigmoidSigmoidgru_cell/add:z:0*
T0*'
_output_shapes
:          }
gru_cell/add_1AddV2gru_cell/split:output:1gru_cell/split_1:output:1*
T0*'
_output_shapes
:          c
gru_cell/Sigmoid_1Sigmoidgru_cell/add_1:z:0*
T0*'
_output_shapes
:          x
gru_cell/mulMulgru_cell/Sigmoid_1:y:0gru_cell/split_1:output:2*
T0*'
_output_shapes
:          t
gru_cell/add_2AddV2gru_cell/split:output:2gru_cell/mul:z:0*
T0*'
_output_shapes
:          [
gru_cell/TanhTanhgru_cell/add_2:z:0*
T0*'
_output_shapes
:          m
gru_cell/mul_1Mulgru_cell/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:          S
gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?t
gru_cell/subSubgru_cell/sub/x:output:0gru_cell/Sigmoid:y:0*
T0*'
_output_shapes
:          l
gru_cell/mul_2Mulgru_cell/sub:z:0gru_cell/Tanh:y:0*
T0*'
_output_shapes
:          q
gru_cell/add_3AddV2gru_cell/mul_1:z:0gru_cell/mul_2:z:0*
T0*'
_output_shapes
:          n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :┼
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥F
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
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ┼
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+gru_cell_readvariableop_gru_3_gru_cell_bias4gru_cell_matmul_readvariableop_gru_3_gru_cell_kernel@gru_cell_matmul_1_readvariableop_gru_3_gru_cell_recurrent_kernel*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :          : : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_22108*
condR
while_cond_22107*8
output_shapes'
%: : : : :          : : : : : *
parallel_iterations Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        ╓
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:          *
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:З
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:          *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ц
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:          [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:          И
NoOpNoOp^gru_cell/MatMul/ReadVariableOp!^gru_cell/MatMul_1/ReadVariableOp^gru_cell/ReadVariableOp^while*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:         │: : : 2@
gru_cell/MatMul/ReadVariableOpgru_cell/MatMul/ReadVariableOp2D
 gru_cell/MatMul_1/ReadVariableOp gru_cell/MatMul_1/ReadVariableOp22
gru_cell/ReadVariableOpgru_cell/ReadVariableOp2
whilewhile:?;
9
_user_specified_name!gru_3/gru_cell/recurrent_kernel:51
/
_user_specified_namegru_3/gru_cell/kernel:3/
-
_user_specified_namegru_3/gru_cell/bias:T P
,
_output_shapes
:         │
 
_user_specified_nameinputs
я	
у
%__inference_gru_3_layer_call_fn_21562
inputs_0%
gru_3_gru_cell_bias:`'
gru_3_gru_cell_kernel:`1
gru_3_gru_cell_recurrent_kernel: `
identityИвStatefulPartitionedCallТ
StatefulPartitionedCallStatefulPartitionedCallinputs_0gru_3_gru_cell_biasgru_3_gru_cell_kernelgru_3_gru_cell_recurrent_kernel*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:          *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *I
fDRB
@__inference_gru_3_layer_call_and_return_conditional_losses_20941o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:          <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  : : : 22
StatefulPartitionedCallStatefulPartitionedCall:?;
9
_user_specified_name!gru_3/gru_cell/recurrent_kernel:51
/
_user_specified_namegru_3/gru_cell/kernel:3/
-
_user_specified_namegru_3/gru_cell/bias:^ Z
4
_output_shapes"
 :                  
"
_user_specified_name
inputs_0
ш
c
G__inference_activation_3_layer_call_and_return_conditional_losses_21008

inputs
identityR
SoftmaxSoftmaxinputs*
T0*-
_output_shapes
:         ││_
IdentityIdentitySoftmax:softmax:0*
T0*-
_output_shapes
:         ││"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:         ││:U Q
-
_output_shapes
:         ││
 
_user_specified_nameinputs
┘9
║
while_body_21798
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
strided_slice_1_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0?
-gru_cell_readvariableop_gru_3_gru_cell_bias_0:`H
6gru_cell_matmul_readvariableop_gru_3_gru_cell_kernel_0:`T
Bgru_cell_matmul_1_readvariableop_gru_3_gru_cell_recurrent_kernel_0: `
identity

identity_1

identity_2

identity_3

identity_4
strided_slice_1O
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor=
+gru_cell_readvariableop_gru_3_gru_cell_bias:`F
4gru_cell_matmul_readvariableop_gru_3_gru_cell_kernel:`R
@gru_cell_matmul_1_readvariableop_gru_3_gru_cell_recurrent_kernel: `Ивgru_cell/MatMul/ReadVariableOpв gru_cell/MatMul_1/ReadVariableOpвgru_cell/ReadVariableOpВ
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       О
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0Е
gru_cell/ReadVariableOpReadVariableOp-gru_cell_readvariableop_gru_3_gru_cell_bias_0*
_output_shapes

:`*
dtype0q
gru_cell/unstackUnpackgru_cell/ReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
numХ
gru_cell/MatMul/ReadVariableOpReadVariableOp6gru_cell_matmul_readvariableop_gru_3_gru_cell_kernel_0*
_output_shapes

:`*
dtype0Я
gru_cell/MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0&gru_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         `Г
gru_cell/BiasAddBiasAddgru_cell/MatMul:product:0gru_cell/unstack:output:0*
T0*'
_output_shapes
:         `c
gru_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         ╛
gru_cell/splitSplit!gru_cell/split/split_dim:output:0gru_cell/BiasAdd:output:0*
T0*M
_output_shapes;
9:          :          :          *
	num_splitг
 gru_cell/MatMul_1/ReadVariableOpReadVariableOpBgru_cell_matmul_1_readvariableop_gru_3_gru_cell_recurrent_kernel_0*
_output_shapes

: `*
dtype0Ж
gru_cell/MatMul_1MatMulplaceholder_2(gru_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         `З
gru_cell/BiasAdd_1BiasAddgru_cell/MatMul_1:product:0gru_cell/unstack:output:1*
T0*'
_output_shapes
:         `c
gru_cell/ConstConst*
_output_shapes
:*
dtype0*!
valueB"            e
gru_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         ъ
gru_cell/split_1SplitVgru_cell/BiasAdd_1:output:0gru_cell/Const:output:0#gru_cell/split_1/split_dim:output:0*

Tlen0*
T0*M
_output_shapes;
9:          :          :          *
	num_split{
gru_cell/addAddV2gru_cell/split:output:0gru_cell/split_1:output:0*
T0*'
_output_shapes
:          _
gru_cell/SigmoidSigmoidgru_cell/add:z:0*
T0*'
_output_shapes
:          }
gru_cell/add_1AddV2gru_cell/split:output:1gru_cell/split_1:output:1*
T0*'
_output_shapes
:          c
gru_cell/Sigmoid_1Sigmoidgru_cell/add_1:z:0*
T0*'
_output_shapes
:          x
gru_cell/mulMulgru_cell/Sigmoid_1:y:0gru_cell/split_1:output:2*
T0*'
_output_shapes
:          t
gru_cell/add_2AddV2gru_cell/split:output:2gru_cell/mul:z:0*
T0*'
_output_shapes
:          [
gru_cell/TanhTanhgru_cell/add_2:z:0*
T0*'
_output_shapes
:          l
gru_cell/mul_1Mulgru_cell/Sigmoid:y:0placeholder_2*
T0*'
_output_shapes
:          S
gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?t
gru_cell/subSubgru_cell/sub/x:output:0gru_cell/Sigmoid:y:0*
T0*'
_output_shapes
:          l
gru_cell/mul_2Mulgru_cell/sub:z:0gru_cell/Tanh:y:0*
T0*'
_output_shapes
:          q
gru_cell/add_3AddV2gru_cell/mul_1:z:0gru_cell/mul_2:z:0*
T0*'
_output_shapes
:          l
*TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : ╤
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_13TensorArrayV2Write/TensorListSetItem/index:output:0gru_cell/add_3:z:0*
_output_shapes
: *
element_dtype0:щш╥G
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
:          А
NoOpNoOp^gru_cell/MatMul/ReadVariableOp!^gru_cell/MatMul_1/ReadVariableOp^gru_cell/ReadVariableOp*
_output_shapes
 "Ж
@gru_cell_matmul_1_readvariableop_gru_3_gru_cell_recurrent_kernelBgru_cell_matmul_1_readvariableop_gru_3_gru_cell_recurrent_kernel_0"n
4gru_cell_matmul_readvariableop_gru_3_gru_cell_kernel6gru_cell_matmul_readvariableop_gru_3_gru_cell_kernel_0"\
+gru_cell_readvariableop_gru_3_gru_cell_bias-gru_cell_readvariableop_gru_3_gru_cell_bias_0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
identityIdentity:output:0"$
strided_slice_1strided_slice_1_0"Ь
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :          : : : : : 2@
gru_cell/MatMul/ReadVariableOpgru_cell/MatMul/ReadVariableOp2D
 gru_cell/MatMul_1/ReadVariableOp gru_cell/MatMul_1/ReadVariableOp22
gru_cell/ReadVariableOpgru_cell/ReadVariableOp:?	;
9
_user_specified_name!gru_3/gru_cell/recurrent_kernel:51
/
_user_specified_namegru_3/gru_cell/kernel:3/
-
_user_specified_namegru_3/gru_cell/bias:_[
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
:          :
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
Х"
С
C__inference_model_22_layer_call_and_return_conditional_losses_21379
input_prior+
gru_3_gru_3_gru_cell_bias:`-
gru_3_gru_3_gru_cell_kernel:`7
%gru_3_gru_3_gru_cell_recurrent_kernel: `*
dense_19_dense_19_kernel: $
dense_19_dense_19_bias:
identityИв dense_19/StatefulPartitionedCallвgru_3/StatefulPartitionedCall╦
masking_prior/PartitionedCallPartitionedCallinput_prior*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         │
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_masking_prior_layer_call_and_return_conditional_losses_20992А
dot_6/PartitionedCallPartitionedCall&masking_prior/PartitionedCall:output:0&masking_prior/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:         ││* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *I
fDRB
@__inference_dot_6_layer_call_and_return_conditional_losses_21002▌
activation_3/PartitionedCallPartitionedCalldot_6/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:         ││* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_activation_3_layer_call_and_return_conditional_losses_21008■
dot_7/PartitionedCallPartitionedCall%activation_3/PartitionedCall:output:0&masking_prior/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         │
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *I
fDRB
@__inference_dot_7_layer_call_and_return_conditional_losses_21016Б
multiply_3/PartitionedCallPartitionedCalldot_7/PartitionedCall:output:0&masking_prior/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         │
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_multiply_3_layer_call_and_return_conditional_losses_21023М
concatenate_3/PartitionedCallPartitionedCall&masking_prior/PartitionedCall:output:0#multiply_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         │* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_concatenate_3_layer_call_and_return_conditional_losses_21031╚
gru_3/StatefulPartitionedCallStatefulPartitionedCall&concatenate_3/PartitionedCall:output:0gru_3_gru_3_gru_cell_biasgru_3_gru_3_gru_cell_kernel%gru_3_gru_3_gru_cell_recurrent_kernel*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:          *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *I
fDRB
@__inference_gru_3_layer_call_and_return_conditional_losses_21370Я
 dense_19/StatefulPartitionedCallStatefulPartitionedCall&gru_3/StatefulPartitionedCall:output:0dense_19_dense_19_kerneldense_19_dense_19_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_dense_19_layer_call_and_return_conditional_losses_21202x
IdentityIdentity)dense_19/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         e
NoOpNoOp!^dense_19/StatefulPartitionedCall^gru_3/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":         │
: : : : : 2D
 dense_19/StatefulPartitionedCall dense_19/StatefulPartitionedCall2>
gru_3/StatefulPartitionedCallgru_3/StatefulPartitionedCall:-)
'
_user_specified_namedense_19/bias:/+
)
_user_specified_namedense_19/kernel:?;
9
_user_specified_name!gru_3/gru_cell/recurrent_kernel:51
/
_user_specified_namegru_3/gru_cell/kernel:3/
-
_user_specified_namegru_3/gru_cell/bias:Y U
,
_output_shapes
:         │

%
_user_specified_nameinput_prior
╢
й
(__inference_model_22_layer_call_fn_21399
input_prior%
gru_3_gru_cell_bias:`'
gru_3_gru_cell_kernel:`1
gru_3_gru_cell_recurrent_kernel: `!
dense_19_kernel: 
dense_19_bias:
identityИвStatefulPartitionedCall╝
StatefulPartitionedCallStatefulPartitionedCallinput_priorgru_3_gru_cell_biasgru_3_gru_cell_kernelgru_3_gru_cell_recurrent_kerneldense_19_kerneldense_19_bias*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_model_22_layer_call_and_return_conditional_losses_21379o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":         │
: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:-)
'
_user_specified_namedense_19/bias:/+
)
_user_specified_namedense_19/kernel:?;
9
_user_specified_name!gru_3/gru_cell/recurrent_kernel:51
/
_user_specified_namegru_3/gru_cell/kernel:3/
-
_user_specified_namegru_3/gru_cell/bias:Y U
,
_output_shapes
:         │

%
_user_specified_nameinput_prior
Ў
у
while_cond_20878
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
less_strided_slice_1-
)while_cond_20878___redundant_placeholder0-
)while_cond_20878___redundant_placeholder1-
)while_cond_20878___redundant_placeholder2-
)while_cond_20878___redundant_placeholder3
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
-: : : : :          : :::::
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
:          :
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
▓N
а
@__inference_gru_3_layer_call_and_return_conditional_losses_21187

inputs=
+gru_cell_readvariableop_gru_3_gru_cell_bias:`F
4gru_cell_matmul_readvariableop_gru_3_gru_cell_kernel:`R
@gru_cell_matmul_1_readvariableop_gru_3_gru_cell_recurrent_kernel: `
identityИвgru_cell/MatMul/ReadVariableOpв gru_cell/MatMul_1/ReadVariableOpвgru_cell/ReadVariableOpвwhileI
ShapeShapeinputs*
T0*
_output_shapes
::э╧]
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
valueB:╤
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
:          c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          n
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:│         R
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
::э╧_
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
valueB:█
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
         ┤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥Ж
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       р
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥_
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
:         *
shrink_axis_maskГ
gru_cell/ReadVariableOpReadVariableOp+gru_cell_readvariableop_gru_3_gru_cell_bias*
_output_shapes

:`*
dtype0q
gru_cell/unstackUnpackgru_cell/ReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
numУ
gru_cell/MatMul/ReadVariableOpReadVariableOp4gru_cell_matmul_readvariableop_gru_3_gru_cell_kernel*
_output_shapes

:`*
dtype0Н
gru_cell/MatMulMatMulstrided_slice_2:output:0&gru_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         `Г
gru_cell/BiasAddBiasAddgru_cell/MatMul:product:0gru_cell/unstack:output:0*
T0*'
_output_shapes
:         `c
gru_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         ╛
gru_cell/splitSplit!gru_cell/split/split_dim:output:0gru_cell/BiasAdd:output:0*
T0*M
_output_shapes;
9:          :          :          *
	num_splitб
 gru_cell/MatMul_1/ReadVariableOpReadVariableOp@gru_cell_matmul_1_readvariableop_gru_3_gru_cell_recurrent_kernel*
_output_shapes

: `*
dtype0З
gru_cell/MatMul_1MatMulzeros:output:0(gru_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         `З
gru_cell/BiasAdd_1BiasAddgru_cell/MatMul_1:product:0gru_cell/unstack:output:1*
T0*'
_output_shapes
:         `c
gru_cell/ConstConst*
_output_shapes
:*
dtype0*!
valueB"            e
gru_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         ъ
gru_cell/split_1SplitVgru_cell/BiasAdd_1:output:0gru_cell/Const:output:0#gru_cell/split_1/split_dim:output:0*

Tlen0*
T0*M
_output_shapes;
9:          :          :          *
	num_split{
gru_cell/addAddV2gru_cell/split:output:0gru_cell/split_1:output:0*
T0*'
_output_shapes
:          _
gru_cell/SigmoidSigmoidgru_cell/add:z:0*
T0*'
_output_shapes
:          }
gru_cell/add_1AddV2gru_cell/split:output:1gru_cell/split_1:output:1*
T0*'
_output_shapes
:          c
gru_cell/Sigmoid_1Sigmoidgru_cell/add_1:z:0*
T0*'
_output_shapes
:          x
gru_cell/mulMulgru_cell/Sigmoid_1:y:0gru_cell/split_1:output:2*
T0*'
_output_shapes
:          t
gru_cell/add_2AddV2gru_cell/split:output:2gru_cell/mul:z:0*
T0*'
_output_shapes
:          [
gru_cell/TanhTanhgru_cell/add_2:z:0*
T0*'
_output_shapes
:          m
gru_cell/mul_1Mulgru_cell/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:          S
gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?t
gru_cell/subSubgru_cell/sub/x:output:0gru_cell/Sigmoid:y:0*
T0*'
_output_shapes
:          l
gru_cell/mul_2Mulgru_cell/sub:z:0gru_cell/Tanh:y:0*
T0*'
_output_shapes
:          q
gru_cell/add_3AddV2gru_cell/mul_1:z:0gru_cell/mul_2:z:0*
T0*'
_output_shapes
:          n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :┼
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥F
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
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ┼
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+gru_cell_readvariableop_gru_3_gru_cell_bias4gru_cell_matmul_readvariableop_gru_3_gru_cell_kernel@gru_cell_matmul_1_readvariableop_gru_3_gru_cell_recurrent_kernel*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :          : : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_21097*
condR
while_cond_21096*8
output_shapes'
%: : : : :          : : : : : *
parallel_iterations Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        ╓
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:          *
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:З
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:          *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ц
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:          [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:          И
NoOpNoOp^gru_cell/MatMul/ReadVariableOp!^gru_cell/MatMul_1/ReadVariableOp^gru_cell/ReadVariableOp^while*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:         │: : : 2@
gru_cell/MatMul/ReadVariableOpgru_cell/MatMul/ReadVariableOp2D
 gru_cell/MatMul_1/ReadVariableOp gru_cell/MatMul_1/ReadVariableOp22
gru_cell/ReadVariableOpgru_cell/ReadVariableOp2
whilewhile:?;
9
_user_specified_name!gru_3/gru_cell/recurrent_kernel:51
/
_user_specified_namegru_3/gru_cell/kernel:3/
-
_user_specified_namegru_3/gru_cell/bias:T P
,
_output_shapes
:         │
 
_user_specified_nameinputs
╪
Б
C__inference_gru_cell_layer_call_and_return_conditional_losses_20730

inputs

states4
"readvariableop_gru_3_gru_cell_bias:`=
+matmul_readvariableop_gru_3_gru_cell_kernel:`I
7matmul_1_readvariableop_gru_3_gru_cell_recurrent_kernel: `
identity

identity_1ИвMatMul/ReadVariableOpвMatMul_1/ReadVariableOpвReadVariableOpq
ReadVariableOpReadVariableOp"readvariableop_gru_3_gru_cell_bias*
_output_shapes

:`*
dtype0_
unstackUnpackReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
numБ
MatMul/ReadVariableOpReadVariableOp+matmul_readvariableop_gru_3_gru_cell_kernel*
_output_shapes

:`*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         `h
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*'
_output_shapes
:         `Z
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         г
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*M
_output_shapes;
9:          :          :          *
	num_splitП
MatMul_1/ReadVariableOpReadVariableOp7matmul_1_readvariableop_gru_3_gru_cell_recurrent_kernel*
_output_shapes

: `*
dtype0m
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         `l
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*'
_output_shapes
:         `Z
ConstConst*
_output_shapes
:*
dtype0*!
valueB"            \
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         ╞
split_1SplitVBiasAdd_1:output:0Const:output:0split_1/split_dim:output:0*

Tlen0*
T0*M
_output_shapes;
9:          :          :          *
	num_split`
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:          M
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:          b
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:          Q
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:          ]
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:          Y
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:          I
TanhTanh	add_2:z:0*
T0*'
_output_shapes
:          S
mul_1MulSigmoid:y:0states*
T0*'
_output_shapes
:          J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:          Q
mul_2Mulsub:z:0Tanh:y:0*
T0*'
_output_shapes
:          V
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*'
_output_shapes
:          X
IdentityIdentity	add_3:z:0^NoOp*
T0*'
_output_shapes
:          Z

Identity_1Identity	add_3:z:0^NoOp*
T0*'
_output_shapes
:          e
NoOpNoOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
_output_shapes
 "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:         :          : : : 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp:?;
9
_user_specified_name!gru_3/gru_cell/recurrent_kernel:51
/
_user_specified_namegru_3/gru_cell/kernel:3/
-
_user_specified_namegru_3/gru_cell/bias:OK
'
_output_shapes
:          
 
_user_specified_namestates:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
╢
й
(__inference_model_22_layer_call_fn_21389
input_prior%
gru_3_gru_cell_bias:`'
gru_3_gru_cell_kernel:`1
gru_3_gru_cell_recurrent_kernel: `!
dense_19_kernel: 
dense_19_bias:
identityИвStatefulPartitionedCall╝
StatefulPartitionedCallStatefulPartitionedCallinput_priorgru_3_gru_cell_biasgru_3_gru_cell_kernelgru_3_gru_cell_recurrent_kerneldense_19_kerneldense_19_bias*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_model_22_layer_call_and_return_conditional_losses_21207o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":         │
: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:-)
'
_user_specified_namedense_19/bias:/+
)
_user_specified_namedense_19/kernel:?;
9
_user_specified_name!gru_3/gru_cell/recurrent_kernel:51
/
_user_specified_namegru_3/gru_cell/kernel:3/
-
_user_specified_namegru_3/gru_cell/bias:Y U
,
_output_shapes
:         │

%
_user_specified_nameinput_prior
О
В
(__inference_gru_cell_layer_call_fn_22238

inputs
states_0%
gru_3_gru_cell_bias:`'
gru_3_gru_cell_kernel:`1
gru_3_gru_cell_recurrent_kernel: `
identity

identity_1ИвStatefulPartitionedCall▓
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0gru_3_gru_cell_biasgru_3_gru_cell_kernelgru_3_gru_cell_recurrent_kernel*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:          :          *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_gru_cell_layer_call_and_return_conditional_losses_20868o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:          q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:          <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:         :          : : : 22
StatefulPartitionedCallStatefulPartitionedCall:?;
9
_user_specified_name!gru_3/gru_cell/recurrent_kernel:51
/
_user_specified_namegru_3/gru_cell/kernel:3/
-
_user_specified_namegru_3/gru_cell/bias:QM
'
_output_shapes
:          
"
_user_specified_name
states_0:O K
'
_output_shapes
:         
 
_user_specified_nameinputs"эL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*╕
serving_defaultд
H
input_prior9
serving_default_input_prior:0         │
<
dense_190
StatefulPartitionedCall:0         tensorflow/serving/predict:Мы
М
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
е
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
п
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
axes"
_tf_keras_layer
е
 	variables
!trainable_variables
"regularization_losses
#	keras_api
$__call__
*%&call_and_return_all_conditional_losses"
_tf_keras_layer
п
&	variables
'trainable_variables
(regularization_losses
)	keras_api
*__call__
*+&call_and_return_all_conditional_losses
,axes"
_tf_keras_layer
е
-	variables
.trainable_variables
/regularization_losses
0	keras_api
1__call__
*2&call_and_return_all_conditional_losses"
_tf_keras_layer
е
3	variables
4trainable_variables
5regularization_losses
6	keras_api
7__call__
*8&call_and_return_all_conditional_losses"
_tf_keras_layer
┌
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
╗
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
╩
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
├
Rtrace_0
Strace_12М
(__inference_model_22_layer_call_fn_21389
(__inference_model_22_layer_call_fn_21399╡
о▓к
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsв
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zRtrace_0zStrace_1
∙
Ttrace_0
Utrace_12┬
C__inference_model_22_layer_call_and_return_conditional_losses_21207
C__inference_model_22_layer_call_and_return_conditional_losses_21379╡
о▓к
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsв
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zTtrace_0zUtrace_1
╧B╠
 __inference__wrapped_model_20665input_prior"Ш
С▓Н
FullArgSpec
argsЪ

jargs_0
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╖
Viter

Wbeta_1

Xbeta_2
	Ydecay
Zlearning_rateHm╕Im╣Jm║Km╗Lm╝Hv╜Iv╛Jv┐Kv└Lv┴"
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
н
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
atrace_02╩
-__inference_masking_prior_layer_call_fn_21472Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zatrace_0
В
btrace_02х
H__inference_masking_prior_layer_call_and_return_conditional_losses_21483Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zbtrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
н
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
▀
htrace_02┬
%__inference_dot_6_layer_call_fn_21489Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zhtrace_0
·
itrace_02▌
@__inference_dot_6_layer_call_and_return_conditional_losses_21498Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zitrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
н
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
otrace_02╔
,__inference_activation_3_layer_call_fn_21503Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zotrace_0
Б
ptrace_02ф
G__inference_activation_3_layer_call_and_return_conditional_losses_21508Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zptrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
н
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
▀
vtrace_02┬
%__inference_dot_7_layer_call_fn_21514Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zvtrace_0
·
wtrace_02▌
@__inference_dot_7_layer_call_and_return_conditional_losses_21521Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zwtrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
н
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
}trace_02╟
*__inference_multiply_3_layer_call_fn_21527Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z}trace_0
 
~trace_02т
E__inference_multiply_3_layer_call_and_return_conditional_losses_21533Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z~trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
▒
non_trainable_variables
Аlayers
Бmetrics
 Вlayer_regularization_losses
Гlayer_metrics
3	variables
4trainable_variables
5regularization_losses
7__call__
*8&call_and_return_all_conditional_losses
&8"call_and_return_conditional_losses"
_generic_user_object
щ
Дtrace_02╩
-__inference_concatenate_3_layer_call_fn_21539Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zДtrace_0
Д
Еtrace_02х
H__inference_concatenate_3_layer_call_and_return_conditional_losses_21546Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zЕtrace_0
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
┐
Жstates
Зnon_trainable_variables
Иlayers
Йmetrics
 Кlayer_regularization_losses
Лlayer_metrics
9	variables
:trainable_variables
;regularization_losses
=__call__
*>&call_and_return_all_conditional_losses
&>"call_and_return_conditional_losses"
_generic_user_object
▄
Мtrace_0
Нtrace_1
Оtrace_2
Пtrace_32щ
%__inference_gru_3_layer_call_fn_21554
%__inference_gru_3_layer_call_fn_21562
%__inference_gru_3_layer_call_fn_21570
%__inference_gru_3_layer_call_fn_21578╩
├▓┐
FullArgSpec:
args2Ъ/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsв

 
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zМtrace_0zНtrace_1zОtrace_2zПtrace_3
╚
Рtrace_0
Сtrace_1
Тtrace_2
Уtrace_32╒
@__inference_gru_3_layer_call_and_return_conditional_losses_21733
@__inference_gru_3_layer_call_and_return_conditional_losses_21888
@__inference_gru_3_layer_call_and_return_conditional_losses_22043
@__inference_gru_3_layer_call_and_return_conditional_losses_22198╩
├▓┐
FullArgSpec:
args2Ъ/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsв

 
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zРtrace_0zСtrace_1zТtrace_2zУtrace_3
"
_generic_user_object
я
Ф	variables
Хtrainable_variables
Цregularization_losses
Ч	keras_api
Ш__call__
+Щ&call_and_return_all_conditional_losses
Ъ_random_generator

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
▓
Ыnon_trainable_variables
Ьlayers
Эmetrics
 Юlayer_regularization_losses
Яlayer_metrics
B	variables
Ctrainable_variables
Dregularization_losses
F__call__
*G&call_and_return_all_conditional_losses
&G"call_and_return_conditional_losses"
_generic_user_object
ф
аtrace_02┼
(__inference_dense_19_layer_call_fn_22205Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zаtrace_0
 
бtrace_02р
C__inference_dense_19_layer_call_and_return_conditional_losses_22216Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zбtrace_0
!: 2dense_19/kernel
:2dense_19/bias
':%`2gru_3/gru_cell/kernel
1:/ `2gru_3/gru_cell/recurrent_kernel
%:#`2gru_3/gru_cell/bias
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
в0
г1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ыBш
(__inference_model_22_layer_call_fn_21389input_prior"м
е▓б
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ыBш
(__inference_model_22_layer_call_fn_21399input_prior"м
е▓б
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЖBГ
C__inference_model_22_layer_call_and_return_conditional_losses_21207input_prior"м
е▓б
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЖBГ
C__inference_model_22_layer_call_and_return_conditional_losses_21379input_prior"м
е▓б
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
:	 (2training_6/Adam/iter
 : (2training_6/Adam/beta_1
 : (2training_6/Adam/beta_2
: (2training_6/Adam/decay
':% (2training_6/Adam/learning_rate
╫B╘
#__inference_signature_wrapper_21467input_prior"Э
Ц▓Т
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
  

kwonlyargsЪ
jinput_prior
kwonlydefaults
 
annotationsк *
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
╫B╘
-__inference_masking_prior_layer_call_fn_21472inputs"Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЄBя
H__inference_masking_prior_layer_call_and_return_conditional_losses_21483inputs"Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
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
█B╪
%__inference_dot_6_layer_call_fn_21489inputs_0inputs_1"Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЎBє
@__inference_dot_6_layer_call_and_return_conditional_losses_21498inputs_0inputs_1"Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
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
╓B╙
,__inference_activation_3_layer_call_fn_21503inputs"Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ёBю
G__inference_activation_3_layer_call_and_return_conditional_losses_21508inputs"Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
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
█B╪
%__inference_dot_7_layer_call_fn_21514inputs_0inputs_1"Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЎBє
@__inference_dot_7_layer_call_and_return_conditional_losses_21521inputs_0inputs_1"Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
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
рB▌
*__inference_multiply_3_layer_call_fn_21527inputs_0inputs_1"Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
√B°
E__inference_multiply_3_layer_call_and_return_conditional_losses_21533inputs_0inputs_1"Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
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
-__inference_concatenate_3_layer_call_fn_21539inputs_0inputs_1"Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
■B√
H__inference_concatenate_3_layer_call_and_return_conditional_losses_21546inputs_0inputs_1"Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
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
ЎBє
%__inference_gru_3_layer_call_fn_21554inputs_0"╜
╢▓▓
FullArgSpec:
args2Ъ/
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

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЎBє
%__inference_gru_3_layer_call_fn_21562inputs_0"╜
╢▓▓
FullArgSpec:
args2Ъ/
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

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЇBё
%__inference_gru_3_layer_call_fn_21570inputs"╜
╢▓▓
FullArgSpec:
args2Ъ/
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

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЇBё
%__inference_gru_3_layer_call_fn_21578inputs"╜
╢▓▓
FullArgSpec:
args2Ъ/
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

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
СBО
@__inference_gru_3_layer_call_and_return_conditional_losses_21733inputs_0"╜
╢▓▓
FullArgSpec:
args2Ъ/
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

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
СBО
@__inference_gru_3_layer_call_and_return_conditional_losses_21888inputs_0"╜
╢▓▓
FullArgSpec:
args2Ъ/
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

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ПBМ
@__inference_gru_3_layer_call_and_return_conditional_losses_22043inputs"╜
╢▓▓
FullArgSpec:
args2Ъ/
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

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ПBМ
@__inference_gru_3_layer_call_and_return_conditional_losses_22198inputs"╜
╢▓▓
FullArgSpec:
args2Ъ/
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

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
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
╕
дnon_trainable_variables
еlayers
жmetrics
 зlayer_regularization_losses
иlayer_metrics
Ф	variables
Хtrainable_variables
Цregularization_losses
Ш__call__
+Щ&call_and_return_all_conditional_losses
'Щ"call_and_return_conditional_losses"
_generic_user_object
┼
йtrace_0
кtrace_12К
(__inference_gru_cell_layer_call_fn_22227
(__inference_gru_cell_layer_call_fn_22238│
м▓и
FullArgSpec+
args#Ъ 
jinputs
jstates

jtraining
varargs
 
varkw
 
defaultsв
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zйtrace_0zкtrace_1
√
лtrace_0
мtrace_12└
C__inference_gru_cell_layer_call_and_return_conditional_losses_22277
C__inference_gru_cell_layer_call_and_return_conditional_losses_22316│
м▓и
FullArgSpec+
args#Ъ 
jinputs
jstates

jtraining
varargs
 
varkw
 
defaultsв
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zлtrace_0zмtrace_1
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
╥B╧
(__inference_dense_19_layer_call_fn_22205inputs"Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
эBъ
C__inference_dense_19_layer_call_and_return_conditional_losses_22216inputs"Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
c
н	variables
о	keras_api

пtotal

░count
▒
_fn_kwargs"
_tf_keras_metric
Р
▓	variables
│	keras_api
┤true_positives
╡true_negatives
╢false_positives
╖false_negatives"
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
ЄBя
(__inference_gru_cell_layer_call_fn_22227inputsstates_0"о
з▓г
FullArgSpec+
args#Ъ 
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЄBя
(__inference_gru_cell_layer_call_fn_22238inputsstates_0"о
з▓г
FullArgSpec+
args#Ъ 
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
НBК
C__inference_gru_cell_layer_call_and_return_conditional_losses_22277inputsstates_0"о
з▓г
FullArgSpec+
args#Ъ 
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
НBК
C__inference_gru_cell_layer_call_and_return_conditional_losses_22316inputsstates_0"о
з▓г
FullArgSpec+
args#Ъ 
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
0
п0
░1"
trackable_list_wrapper
.
н	variables"
_generic_user_object
:  (2total_3
:  (2count_3
 "
trackable_dict_wrapper
@
┤0
╡1
╢2
╖3"
trackable_list_wrapper
.
▓	variables"
_generic_user_object
!:╚ (2true_positives_3
!:╚ (2true_negatives_3
": ╚ (2false_positives_3
": ╚ (2false_negatives_3
1:/ 2!training_6/Adam/dense_19/kernel/m
+:)2training_6/Adam/dense_19/bias/m
7:5`2'training_6/Adam/gru_3/gru_cell/kernel/m
A:? `21training_6/Adam/gru_3/gru_cell/recurrent_kernel/m
5:3`2%training_6/Adam/gru_3/gru_cell/bias/m
1:/ 2!training_6/Adam/dense_19/kernel/v
+:)2training_6/Adam/dense_19/bias/v
7:5`2'training_6/Adam/gru_3/gru_cell/kernel/v
A:? `21training_6/Adam/gru_3/gru_cell/recurrent_kernel/v
5:3`2%training_6/Adam/gru_3/gru_cell/bias/vЫ
 __inference__wrapped_model_20665wLJKHI9в6
/в,
*К'
input_prior         │

к "3к0
.
dense_19"К
dense_19         ╢
G__inference_activation_3_layer_call_and_return_conditional_losses_21508k5в2
+в(
&К#
inputs         ││
к "2в/
(К%
tensor_0         ││
Ъ Р
,__inference_activation_3_layer_call_fn_21503`5в2
+в(
&К#
inputs         ││
к "'К$
unknown         ││ц
H__inference_concatenate_3_layer_call_and_return_conditional_losses_21546Щdвa
ZвW
UЪR
'К$
inputs_0         │

'К$
inputs_1         │

к "1в.
'К$
tensor_0         │
Ъ └
-__inference_concatenate_3_layer_call_fn_21539Оdвa
ZвW
UЪR
'К$
inputs_0         │

'К$
inputs_1         │

к "&К#
unknown         │к
C__inference_dense_19_layer_call_and_return_conditional_losses_22216cHI/в,
%в"
 К
inputs          
к ",в)
"К
tensor_0         
Ъ Д
(__inference_dense_19_layer_call_fn_22205XHI/в,
%в"
 К
inputs          
к "!К
unknown         ▀
@__inference_dot_6_layer_call_and_return_conditional_losses_21498Ъdвa
ZвW
UЪR
'К$
inputs_0         │

'К$
inputs_1         │

к "2в/
(К%
tensor_0         ││
Ъ ╣
%__inference_dot_6_layer_call_fn_21489Пdвa
ZвW
UЪR
'К$
inputs_0         │

'К$
inputs_1         │

к "'К$
unknown         ││▀
@__inference_dot_7_layer_call_and_return_conditional_losses_21521Ъeвb
[вX
VЪS
(К%
inputs_0         ││
'К$
inputs_1         │

к "1в.
'К$
tensor_0         │

Ъ ╣
%__inference_dot_7_layer_call_fn_21514Пeвb
[вX
VЪS
(К%
inputs_0         ││
'К$
inputs_1         │

к "&К#
unknown         │
╔
@__inference_gru_3_layer_call_and_return_conditional_losses_21733ДLJKOвL
EвB
4Ъ1
/К,
inputs_0                  

 
p

 
к ",в)
"К
tensor_0          
Ъ ╔
@__inference_gru_3_layer_call_and_return_conditional_losses_21888ДLJKOвL
EвB
4Ъ1
/К,
inputs_0                  

 
p 

 
к ",в)
"К
tensor_0          
Ъ ╣
@__inference_gru_3_layer_call_and_return_conditional_losses_22043uLJK@в=
6в3
%К"
inputs         │

 
p

 
к ",в)
"К
tensor_0          
Ъ ╣
@__inference_gru_3_layer_call_and_return_conditional_losses_22198uLJK@в=
6в3
%К"
inputs         │

 
p 

 
к ",в)
"К
tensor_0          
Ъ в
%__inference_gru_3_layer_call_fn_21554yLJKOвL
EвB
4Ъ1
/К,
inputs_0                  

 
p

 
к "!К
unknown          в
%__inference_gru_3_layer_call_fn_21562yLJKOвL
EвB
4Ъ1
/К,
inputs_0                  

 
p 

 
к "!К
unknown          У
%__inference_gru_3_layer_call_fn_21570jLJK@в=
6в3
%К"
inputs         │

 
p

 
к "!К
unknown          У
%__inference_gru_3_layer_call_fn_21578jLJK@в=
6в3
%К"
inputs         │

 
p 

 
к "!К
unknown          Н
C__inference_gru_cell_layer_call_and_return_conditional_losses_22277┼LJK\вY
RвO
 К
inputs         
'в$
"К
states_0          
p
к "`в]
VвS
$К!

tensor_0_0          
+Ъ(
&К#
tensor_0_1_0          
Ъ Н
C__inference_gru_cell_layer_call_and_return_conditional_losses_22316┼LJK\вY
RвO
 К
inputs         
'в$
"К
states_0          
p 
к "`в]
VвS
$К!

tensor_0_0          
+Ъ(
&К#
tensor_0_1_0          
Ъ ф
(__inference_gru_cell_layer_call_fn_22227╖LJK\вY
RвO
 К
inputs         
'в$
"К
states_0          
p
к "RвO
"К
tensor_0          
)Ъ&
$К!

tensor_1_0          ф
(__inference_gru_cell_layer_call_fn_22238╖LJK\вY
RвO
 К
inputs         
'в$
"К
states_0          
p 
к "RвO
"К
tensor_0          
)Ъ&
$К!

tensor_1_0          ╡
H__inference_masking_prior_layer_call_and_return_conditional_losses_21483i4в1
*в'
%К"
inputs         │

к "1в.
'К$
tensor_0         │

Ъ П
-__inference_masking_prior_layer_call_fn_21472^4в1
*в'
%К"
inputs         │

к "&К#
unknown         │
┐
C__inference_model_22_layer_call_and_return_conditional_losses_21207xLJKHIAв>
7в4
*К'
input_prior         │

p

 
к ",в)
"К
tensor_0         
Ъ ┐
C__inference_model_22_layer_call_and_return_conditional_losses_21379xLJKHIAв>
7в4
*К'
input_prior         │

p 

 
к ",в)
"К
tensor_0         
Ъ Щ
(__inference_model_22_layer_call_fn_21389mLJKHIAв>
7в4
*К'
input_prior         │

p

 
к "!К
unknown         Щ
(__inference_model_22_layer_call_fn_21399mLJKHIAв>
7в4
*К'
input_prior         │

p 

 
к "!К
unknown         у
E__inference_multiply_3_layer_call_and_return_conditional_losses_21533Щdвa
ZвW
UЪR
'К$
inputs_0         │

'К$
inputs_1         │

к "1в.
'К$
tensor_0         │

Ъ ╜
*__inference_multiply_3_layer_call_fn_21527Оdвa
ZвW
UЪR
'К$
inputs_0         │

'К$
inputs_1         │

к "&К#
unknown         │
о
#__inference_signature_wrapper_21467ЖLJKHIHвE
в 
>к;
9
input_prior*К'
input_prior         │
"3к0
.
dense_19"К
dense_19         