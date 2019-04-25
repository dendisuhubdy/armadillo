// Copyright 2018 Ryan Curtin (http://ratml.org)
// Copyright 2018 National ICT Australia (NICTA)
// 
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
// 
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ------------------------------------------------------------------------



//! \addtogroup spop_vectorise
//! @{


template<typename T1>
inline
void
spop_vectorise_col::apply(SpMat<typename T1::elem_type>& out, const SpOp<T1, spop_vectorise_col>& in)
  {
  arma_extra_debug_sigprint();

  SpProxy<T1> P(in.m);

  spop_vectorise_col::apply_proxy(out, P);
  }


template<typename T1>
inline
void
spop_vectorise_col::apply_proxy(SpMat<typename T1::elem_type>& out, const SpProxy<T1>& P)
  {
  // Check for aliasing.
  if (P.is_alias(out))
    {
    // If it's an alias, it's just a reshape...
    out.reshape(P.get_n_elem(), 1);
    return;
    }

  // It's not an alias---so copy elements over.  It's easiest to just use the
  // iterator regardless.
  out.zeros(P.get_n_elem(), 1);
  typename SpProxy<T1>::const_iterator_type it = P.begin();
  const typename SpProxy<T1>::const_iterator_type end = P.end();
  while (it != end)
    {
    out.at((it.col() * P.get_n_rows()) + it.row(), 0) = (*it);

    ++it;
    }

  out.sync();
  }



template<typename T1>
inline
void
spop_vectorise_row::apply(SpMat<typename T1::elem_type>& out, const SpOp<T1, spop_vectorise_row>& in)
  {
  arma_extra_debug_sigprint();

  const SpProxy<T1> P(in.m);

  spop_vectorise_row::apply_proxy(out, P);
  }


template<typename T1>
inline
void
spop_vectorise_row::apply_proxy(SpMat<typename T1::elem_type>& out, const SpProxy<T1>& P)
  {
  // Check for aliasing.
  bool is_alias = P.is_alias(out);
  SpMat<typename T1::elem_type> tmp;
  SpMat<typename T1::elem_type>& out_alias = (is_alias) ? tmp : out;

  // We have to iterate row-wise.
  typename SpProxy<T1>::const_row_iterator_type it = P.begin_row();
  typename SpProxy<T1>::const_row_iterator_type end = P.end_row();

  out_alias.zeros(1, P.get_n_elem());

  while (it != end)
    {
    out_alias.at(0, it.row() * P.get_n_cols() + it.col()) = (*it);

    ++it;
    }

  out_alias.sync();

  if (is_alias)
    {
    out.steal_mem(tmp);
    }
  }

template<typename T1>
inline
void
spop_vectorise_all::apply(SpMat<typename T1::elem_type>& out, const SpOp<T1, spop_vectorise_all>& in)
  {
  arma_extra_debug_sigprint();

  const SpProxy<T1> P(in.m);

  const uword dim = in.aux_uword_a;

  if (dim == 0)
    {
    spop_vectorise_col::apply_proxy(out, P);
    }
  else
    {
    spop_vectorise_row::apply_proxy(out, P);
    }
  }
//! @}
